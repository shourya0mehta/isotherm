from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DesiredOrder:
    ticker: str
    side: str  # 'bid' = long YES at price_cents ; 'ask' = long NO, quoted at YES price price_cents
    price_cents: int
    count: float
    sleeve: str  # 'A' | 'B'
    fair_yes: float
    reason: str = ""
    event: str = ""
    expires_ts: Optional[int] = None

    @property
    def key(self) -> tuple:
        return (self.ticker, self.side)

    @property
    def unit_cost(self) -> float:
        """Dollars paid per contract if filled."""
        return self.price_cents / 100.0 if self.side == "bid" else (100 - self.price_cents) / 100.0

    @property
    def edge_per_contract(self) -> float:
        p_win = self.fair_yes if self.side == "bid" else 1 - self.fair_yes
        return p_win - self.unit_cost


@dataclass
class Position:
    ticker: str
    event: str
    yes_qty: float = 0.0  # long YES contracts
    yes_cost: float = 0.0  # dollars paid for YES
    no_qty: float = 0.0  # long NO contracts
    no_cost: float = 0.0
    fees: float = 0.0


@dataclass
class Fill:
    ts: float
    ticker: str
    event: str
    side: str
    price_cents: int
    count: float
    sleeve: str
    is_taker: bool = False
    fee: float = 0.0
    order_ref: str = ""
    # paper broker fills only (shadow sizing, Oct 8 review): the order's open size when the trade printed, and the
    # print's contracts that reached it (through our price, or at it beyond the queue ahead); None elsewhere
    intended: Optional[float] = None
    depth: Optional[float] = None


@dataclass
class EventBook:
    """Everything the strategy needs about one city-day ladder (or one hourly event)."""
    event: str
    series: str
    kind: str  # 'high' | 'low' | 'hourly'
    station: str
    local_date: object
    close_ts: float
    markets: list = field(default_factory=list)  # raw market dicts
    brackets: list = field(default_factory=list)  # Bracket
    quotes: dict = field(default_factory=dict)  # ticker -> LadderQuote
    books: dict = field(default_factory=dict)  # ticker -> raw orderbook_fp
