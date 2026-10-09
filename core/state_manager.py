"""Tracks active Kalshi KXBTC15M contracts and merges them with live spot data.

The registry is guarded by a `threading.RLock`. No critical section awaits, so
the async API is safe to call from the event loop while other threads (e.g. a
sync web handler or a backtest worker) read snapshots concurrently.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Protocol

from core.constants import KALSHI_BTC_15M_SERIES, MOMENTUM_WINDOW_SEC
from core.schemas import Btc15mMarketState, MarketQuote, StrategyParameters


class SpotSource(Protocol):
    """Read interface satisfied by `services.btc_spot_feed.SpotPriceBuffer` and `BtcSpotFeed`."""

    def latest_price(self) -> float | None: ...
    def volatility(self, lookback_sec: int) -> float | None: ...
    def momentum(self, window_sec: int) -> float | None: ...
    def is_stale(self, now: datetime | None = None) -> bool: ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


@dataclass(frozen=True)
class ActiveMarket:
    ticker: str
    strike_price: float
    expiration_time: datetime
    order_book_imbalance: float = 0.0
    quote: MarketQuote | None = None


class BtcStateManager:
    """Registry of live KXBTC15M contracts, merged on demand with spot ticks.

    `get_market_state` returns None when the contract is unknown or expired, or
    when spot data is missing, stale, or not yet warm enough for volatility and
    momentum. Callers should treat None as "do not price this market now".
    """

    def __init__(
        self,
        spot: SpotSource,
        params: StrategyParameters | None = None,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._spot = spot
        self._params = params or StrategyParameters()
        self._clock = clock
        self._markets: dict[str, ActiveMarket] = {}
        self._lock = threading.RLock()

    # -- parameters ----------------------------------------------------------

    @property
    def params(self) -> StrategyParameters:
        with self._lock:
            return self._params

    async def update_parameters(self, params: StrategyParameters) -> None:
        with self._lock:
            self._params = params

    # -- market registry -----------------------------------------------------

    async def upsert_market(self, ticker: str, strike_price: float, expiration_time: datetime) -> ActiveMarket:
        if not ticker.startswith(f"{KALSHI_BTC_15M_SERIES}-"):
            raise ValueError(f"{ticker!r} is not a {KALSHI_BTC_15M_SERIES} contract")
        if strike_price <= 0:
            raise ValueError("strike_price must be positive")
        expiration_time = _as_utc(expiration_time)
        with self._lock:
            existing = self._markets.get(ticker)
            market = (
                replace(existing, strike_price=strike_price, expiration_time=expiration_time)
                if existing
                else ActiveMarket(ticker=ticker, strike_price=strike_price, expiration_time=expiration_time)
            )
            self._markets[ticker] = market
            return market

    async def update_quote(self, quote: MarketQuote) -> None:
        with self._lock:
            market = self._require(quote.ticker)
            self._markets[quote.ticker] = replace(market, quote=quote)

    async def update_order_book_imbalance(self, ticker: str, imbalance: float) -> None:
        if not -1.0 <= imbalance <= 1.0:
            raise ValueError("order_book_imbalance must be within [-1.0, 1.0]")
        with self._lock:
            market = self._require(ticker)
            self._markets[ticker] = replace(market, order_book_imbalance=imbalance)

    async def remove_market(self, ticker: str) -> bool:
        with self._lock:
            return self._markets.pop(ticker, None) is not None

    async def prune_expired(self) -> list[str]:
        now = self._clock()
        with self._lock:
            expired = [t for t, m in self._markets.items() if m.expiration_time <= now]
            for t in expired:
                del self._markets[t]
        return expired

    def active_tickers(self) -> list[str]:
        with self._lock:
            return sorted(self._markets)

    def get_market(self, ticker: str) -> ActiveMarket | None:
        with self._lock:
            return self._markets.get(ticker)

    def _require(self, ticker: str) -> ActiveMarket:
        market = self._markets.get(ticker)
        if market is None:
            raise KeyError(f"unknown market {ticker!r}")
        return market

    # -- merged state --------------------------------------------------------

    async def get_market_state(self, ticker: str) -> Btc15mMarketState | None:
        return self.snapshot_market_state(ticker)

    async def get_all_market_states(self) -> list[Btc15mMarketState]:
        return self.snapshot_all_market_states()

    def snapshot_market_state(self, ticker: str) -> Btc15mMarketState | None:
        """Synchronous variant of `get_market_state` for use from worker threads."""
        with self._lock:
            market = self._markets.get(ticker)
            params = self._params
        if market is None:
            return None
        spot = self._spot_inputs(params, self._clock())
        return self._build_state(market, *spot) if spot else None

    def snapshot_all_market_states(self) -> list[Btc15mMarketState]:
        with self._lock:
            markets = sorted(self._markets.values(), key=lambda m: m.expiration_time)
            params = self._params
        now = self._clock()
        spot = self._spot_inputs(params, now)
        if spot is None:
            return []
        states = (self._build_state(m, *spot) for m in markets)
        return [s for s in states if s is not None]

    def _spot_inputs(self, params: StrategyParameters, now: datetime) -> tuple[datetime, float, float, float] | None:
        if self._spot.is_stale(now):
            return None
        price = self._spot.latest_price()
        vol = self._spot.volatility(params.volatility_lookback_sec)
        mom = self._spot.momentum(MOMENTUM_WINDOW_SEC)
        if price is None or vol is None or mom is None:
            return None
        return now, price, vol, mom

    @staticmethod
    def _build_state(
        market: ActiveMarket, now: datetime, price: float, vol: float, mom: float
    ) -> Btc15mMarketState | None:
        seconds_remaining = int((market.expiration_time - now).total_seconds())
        if seconds_remaining <= 0:
            return None
        return Btc15mMarketState(
            ticker=market.ticker,
            strike_price=market.strike_price,
            expiration_time=market.expiration_time,
            seconds_remaining=seconds_remaining,
            current_spot_price=price,
            historical_volatility=vol,
            momentum_1m=mom,
            order_book_imbalance=market.order_book_imbalance,
        )

    # -- background maintenance ----------------------------------------------

    async def run_expiry_pruner(self, interval_sec: float = 1.0) -> None:
        while True:
            await self.prune_expired()
            await asyncio.sleep(interval_sec)
