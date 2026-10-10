"""Align the Coinbase spot feed with what KXBTC15M actually settles on.

Two corrections, both measured from data the agent already records:

* **Index gap.** Contracts settle on the CF Benchmarks BRTI, not Coinbase. Each contract's strike is
  the BRTI averaged over the 60 seconds before its window opens, so `strike - mean(Coinbase over the
  same 60 s)` is one direct observation of the BRTI-minus-Coinbase gap per contract (every 15
  minutes). Across the first 29 contracts it was +$4.58 on average and positive every time, which
  biased the model toward NO whenever spot sat near the strike. The live estimate is the median of
  the most recent observations.
* **Volatility floor.** Five-minute realized volatility read as low as 2-8% annualized in quiet
  stretches, making every near-the-money probability overconfident. A 4-hour realized volatility
  from the persisted 5-second spot history is used as a floor.

Offline on 29 contracts the two together cut the spot model's held-out log loss from 0.578 to 0.519
(market mid: 0.495).
"""

from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from typing import Optional

from core.constants import MOMENTUM_WINDOW_SEC
from services.btc_spot_feed import BtcSpotFeed

CONTRACT_WINDOW = timedelta(minutes=15)
STRIKE_AVERAGE_SEC = 60  # the strike is the BRTI averaged over the minute before the window opens
MIN_PRICES_PER_MINUTE = 8  # persisted spot history is ~one sample per 5 s
MIN_GAP_OBSERVATIONS = 3
MAX_ABS_GAP_USD = 100.0  # anything larger is a data problem, not a basis
MAX_SAMPLE_SPACING_SEC = 30.0  # longer gaps in the history (restarts) are not returns


def _window_mean(times: Sequence[datetime], prices: Sequence[float], end: datetime, seconds: int) -> Optional[float]:
    i, j = bisect_right(times, end - timedelta(seconds=seconds)), bisect_right(times, end)
    if j - i < MIN_PRICES_PER_MINUTE * seconds / 60:
        return None
    return statistics.fmean(prices[i:j])


def gap_observations(
    strikes: Iterable[tuple[datetime, float]], history: Sequence[tuple[datetime, float]]
) -> list[tuple[datetime, float]]:
    """(window open, strike - Coinbase average over the minute before it), oldest first."""
    times = [t for t, _ in history]
    prices = [p for _, p in history]
    out = []
    for close_time, strike in sorted(strikes):
        opened = close_time - CONTRACT_WINDOW
        mean = _window_mean(times, prices, opened, STRIKE_AVERAGE_SEC)
        if mean is not None and abs(strike - mean) <= MAX_ABS_GAP_USD:
            out.append((opened, strike - mean))
    return out


def estimate_index_gap(observations: Sequence[tuple[datetime, float]], recent: int = 8) -> Optional[float]:
    """Median of the most recent gap observations, or None without enough of them."""
    gaps = [g for _, g in observations[-recent:]]
    if len(gaps) < min(MIN_GAP_OBSERVATIONS, recent):
        return None
    return statistics.median(gaps)


def realized_volatility(history: Sequence[tuple[datetime, float]], now: datetime, lookback_sec: int) -> Optional[float]:
    """Annualized volatility from irregularly spaced prices over [now - lookback, now]; None if the
    history covers less than half the window."""
    start = now - timedelta(seconds=lookback_sec)
    pts = [(t, p) for t, p in history if start <= t <= now]
    sum_sq = covered = 0.0
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        dt = (t1 - t0).total_seconds()
        if 0 < dt <= MAX_SAMPLE_SPACING_SEC and p0 > 0 and p1 > 0:
            sum_sq += math.log(p1 / p0) ** 2
            covered += dt
    if covered < lookback_sec / 2 or sum_sq == 0.0:
        return None
    return math.sqrt(sum_sq / covered * 365 * 24 * 3600)


class CalibratedSpot:
    """The spot feed as the pricing models should see it: shifted onto the settlement index, with
    volatility floored by the long-horizon estimate. Satisfies `core.state_manager.SpotSource`."""

    def __init__(self, feed: BtcSpotFeed) -> None:
        self.feed = feed
        self.index_gap = 0.0
        self.long_volatility: Optional[float] = None

    def latest_price(self) -> Optional[float]:
        price = self.feed.latest_price()
        return None if price is None else price + self.index_gap

    def observed_prices(self, lookback_sec: int) -> list[float]:
        """Index-aligned 1-second prices over the last `lookback_sec` seconds."""
        return [p + self.index_gap for p in self.feed.buffer.snapshot(lookback_sec)]

    def volatility(self, lookback_sec: int) -> Optional[float]:
        vol = self.feed.volatility(lookback_sec)
        if vol is None:
            return None  # not warm yet: the floor must not let the agent price early
        return max(vol, self.long_volatility or 0.0)

    def momentum(self, window_sec: int = MOMENTUM_WINDOW_SEC) -> Optional[float]:
        return self.feed.momentum(window_sec)

    def is_stale(self, now: Optional[datetime] = None) -> bool:
        return self.feed.is_stale(now)
