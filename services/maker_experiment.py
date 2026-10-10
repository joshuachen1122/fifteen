"""Resting-order (maker) shadow experiment: would posting a bid have beaten buying at the ask?

Nothing here sends orders. For each LLM-reviewed candidate the agent records the bid it *could*
have rested, then replays Kalshi's public trade tape over the resting period:

* We improve the best bid on our side by 1c when the spread allows (so we are alone at the top
  and any taker selling our side at or below our price hits us first), otherwise we join the
  best bid and only count a fill when a trade prints strictly through our price (we would be
  at the back of the queue).
* A taker "selling our side" shows up on the tape as a taker buying the *other* outcome.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Optional

from execution.kalshi_client import PublicTrade

Side = Literal["yes", "no"]


def maker_limit(side_bid: Optional[int], side_ask: int) -> Optional[tuple[int, bool]]:
    """(limit price, joined_queue) for a resting bid on one side, or None if no sensible bid exists."""
    bid = side_bid if side_bid is not None and side_bid > 0 else 0
    if side_ask <= 1:
        return None
    if side_ask - bid >= 2:
        return max(1, bid + 1), False  # improve: alone at the top of the book
    if bid >= 1:
        return bid, True  # 1c spread: join the best bid
    return None


def side_quotes(side: Side, yes_bid: int, yes_ask: int) -> tuple[Optional[int], int]:
    """(best bid, best ask) on `side`. NO bid = 100 - YES ask, NO ask = 100 - YES bid."""
    if side == "yes":
        return yes_bid, yes_ask
    return (100 - yes_ask if yes_ask < 100 else None), 100 - yes_bid


def first_fill(
    side: Side,
    limit_cents: int,
    joined_queue: bool,
    trades: Sequence[PublicTrade],
    start: datetime,
    end: datetime,
) -> Optional[PublicTrade]:
    """First trade in [start, end] that would have filled a resting bid on `side` at `limit_cents`."""
    for t in trades:
        if t.created_time < start or t.created_time > end or t.taker_side == side:
            continue
        price = t.yes_price_cents if side == "yes" else t.no_price_cents
        if price < limit_cents - 1e-9 or (not joined_queue and price <= limit_cents + 1e-9):
            return t
    return None
