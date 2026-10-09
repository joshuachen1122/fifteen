"""Public BTC-USD spot trade WebSocket feed with a rolling 1-second price buffer.

`SpotPriceBuffer` holds one close price per second for the trailing 15 minutes.
Seconds with no trades are forward-filled with the previous close so the series
is evenly spaced, which keeps the 1-second log-return volatility estimate honest
(a quiet second is a zero return, not a missing one).

All buffer access is guarded by a `threading.Lock`; critical sections never
await, so the buffer is safe to read from the event loop and from worker threads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import numpy as np
import websockets

from core.constants import (
    BINANCE_WS_URL,
    COINBASE_PRODUCT_ID,
    COINBASE_WS_URL,
    MIN_VOLATILITY_SAMPLES,
    MOMENTUM_WINDOW_SEC,
    SECONDS_PER_YEAR,
    SPOT_BUFFER_WINDOW_SEC,
    SPOT_SOURCE_BINANCE,
    SPOT_SOURCE_COINBASE,
    SPOT_STALE_AFTER_SEC,
    VOLATILITY_WINDOW_SEC,
    WS_RECONNECT_INITIAL_SEC,
    WS_RECONNECT_MAX_SEC,
)
from core.schemas import BtcSpotTick

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Exchange message parsers
# ---------------------------------------------------------------------------


def parse_coinbase_message(msg: dict[str, Any]) -> BtcSpotTick | None:
    """Parse a Coinbase Exchange `matches` channel message into a tick."""
    if msg.get("type") not in ("match", "last_match"):
        return None
    if msg.get("product_id", COINBASE_PRODUCT_ID) != COINBASE_PRODUCT_ID:
        return None
    try:
        price = float(msg["price"])
        ts = datetime.fromisoformat(msg["time"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 0:
        return None
    return BtcSpotTick(source=SPOT_SOURCE_COINBASE, price=price, timestamp=_as_utc(ts))


def parse_binance_message(msg: dict[str, Any]) -> BtcSpotTick | None:
    """Parse a Binance `@trade` stream message (raw or combined-stream wrapped)."""
    if "data" in msg and isinstance(msg["data"], dict):
        msg = msg["data"]
    if msg.get("e") != "trade":
        return None
    try:
        price = float(msg["p"])
        ts = datetime.fromtimestamp(int(msg["T"]) / 1000.0, tz=timezone.utc)
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 0:
        return None
    return BtcSpotTick(source=SPOT_SOURCE_BINANCE, price=price, timestamp=ts)


# ---------------------------------------------------------------------------
# Rolling buffer
# ---------------------------------------------------------------------------


class SpotPriceBuffer:
    """Thread-safe rolling buffer of 1-second BTC close prices."""

    def __init__(self, window_sec: int = SPOT_BUFFER_WINDOW_SEC) -> None:
        if window_sec < 2:
            raise ValueError("window_sec must be >= 2")
        self._window_sec = window_sec
        self._prices: deque[float] = deque(maxlen=window_sec)
        self._last_sec: int | None = None
        self._last_tick: BtcSpotTick | None = None
        self._lock = threading.Lock()

    def add_tick(self, tick: BtcSpotTick) -> bool:
        """Record a trade tick. Returns False if the tick was older than the buffer head."""
        ts = _as_utc(tick.timestamp)
        sec = int(ts.timestamp())
        with self._lock:
            if self._last_sec is None:
                self._prices.append(tick.price)
            elif sec == self._last_sec:
                self._prices[-1] = tick.price
            elif sec > self._last_sec:
                gap = sec - self._last_sec - 1
                if gap >= self._window_sec:
                    self._prices.clear()
                else:
                    self._prices.extend([self._prices[-1]] * gap)
                self._prices.append(tick.price)
            else:
                return False
            self._last_sec = sec
            self._last_tick = tick.model_copy(update={"timestamp": ts})
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._prices)

    @property
    def window_sec(self) -> int:
        return self._window_sec

    @property
    def last_tick(self) -> BtcSpotTick | None:
        with self._lock:
            return self._last_tick

    def latest_price(self) -> float | None:
        with self._lock:
            return self._prices[-1] if self._prices else None

    def is_stale(self, now: datetime | None = None, max_age_sec: float = SPOT_STALE_AFTER_SEC) -> bool:
        with self._lock:
            last = self._last_tick
        if last is None:
            return True
        age = ((now or _utc_now()) - last.timestamp).total_seconds()
        return age > max_age_sec

    def snapshot(self, lookback_sec: int | None = None) -> list[float]:
        """Return the trailing `lookback_sec + 1` per-second closes (oldest first)."""
        with self._lock:
            if lookback_sec is None:
                return list(self._prices)
            n = min(lookback_sec + 1, len(self._prices))
            return list(self._prices)[-n:] if n else []

    def volatility(
        self,
        lookback_sec: int = VOLATILITY_WINDOW_SEC,
        min_samples: int = MIN_VOLATILITY_SAMPLES,
    ) -> float | None:
        """Annualized realized volatility from 1-second log returns over `lookback_sec`.

        Returns None until at least `min_samples` returns are available.
        """
        prices = self.snapshot(lookback_sec)
        if len(prices) - 1 < max(min_samples, 2):
            return None
        returns = np.diff(np.log(np.asarray(prices, dtype=np.float64)))
        return float(np.std(returns, ddof=1) * math.sqrt(SECONDS_PER_YEAR))

    def momentum(self, window_sec: int = MOMENTUM_WINDOW_SEC) -> float | None:
        """Price return over the trailing `window_sec`, in percent.

        Returns None until the buffer covers the full window.
        """
        with self._lock:
            if len(self._prices) <= window_sec:
                return None
            now_px = self._prices[-1]
            then_px = self._prices[-1 - window_sec]
        return (now_px / then_px - 1.0) * 100.0


# ---------------------------------------------------------------------------
# WebSocket feed
# ---------------------------------------------------------------------------

_SOURCES: dict[str, tuple[str, Callable[[dict[str, Any]], BtcSpotTick | None]]] = {
    SPOT_SOURCE_COINBASE: (COINBASE_WS_URL, parse_coinbase_message),
    SPOT_SOURCE_BINANCE: (BINANCE_WS_URL, parse_binance_message),
}

TickListener = Callable[[BtcSpotTick], None]


class BtcSpotFeed:
    """Async listener for public BTC-USD trades that maintains a `SpotPriceBuffer`.

    Reconnects with exponential backoff on any connection failure.
    """

    def __init__(
        self,
        source: str = SPOT_SOURCE_COINBASE,
        *,
        url: str | None = None,
        buffer: SpotPriceBuffer | None = None,
    ) -> None:
        if source not in _SOURCES:
            raise ValueError(f"unknown spot source {source!r}; expected one of {sorted(_SOURCES)}")
        default_url, parser = _SOURCES[source]
        self.source = source
        self.url = url or default_url
        self.buffer = buffer or SpotPriceBuffer()
        self._parser = parser
        self._listeners: list[TickListener] = []
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.connected = asyncio.Event()

    # -- lifecycle -----------------------------------------------------------

    def add_listener(self, listener: TickListener) -> None:
        self._listeners.append(listener)

    def start(self) -> asyncio.Task[None]:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self.run(), name=f"btc-spot-feed-{self.source}")
        return self._task

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def run(self) -> None:
        backoff = WS_RECONNECT_INITIAL_SEC
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as ws:
                    await self._subscribe(ws)
                    self.connected.set()
                    backoff = WS_RECONNECT_INITIAL_SEC
                    logger.info("spot feed connected: %s", self.url)
                    async for raw in ws:
                        self.handle_raw_message(raw)
                        if self._stop.is_set():
                            break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - any failure triggers a reconnect
                logger.warning("spot feed %s error: %s; reconnecting in %.1fs", self.source, exc, backoff)
            finally:
                self.connected.clear()
            if self._stop.is_set():
                break
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, WS_RECONNECT_MAX_SEC)

    async def _subscribe(self, ws: Any) -> None:
        if self.source == SPOT_SOURCE_COINBASE:
            await ws.send(
                json.dumps(
                    {"type": "subscribe", "product_ids": [COINBASE_PRODUCT_ID], "channels": ["matches"]}
                )
            )
        # Binance raw streams are subscribed via the URL path.

    # -- message handling ----------------------------------------------------

    def handle_raw_message(self, raw: str | bytes) -> BtcSpotTick | None:
        try:
            msg = json.loads(raw)
        except (TypeError, ValueError):
            return None
        if not isinstance(msg, dict):
            return None
        tick = self._parser(msg)
        if tick is None or not self.buffer.add_tick(tick):
            return None
        for listener in self._listeners:
            try:
                listener(tick)
            except Exception:  # noqa: BLE001 - a bad listener must not kill the feed
                logger.exception("spot tick listener failed")
        return tick

    # -- convenience accessors -----------------------------------------------

    def latest_price(self) -> float | None:
        return self.buffer.latest_price()

    def volatility(self, lookback_sec: int = VOLATILITY_WINDOW_SEC) -> float | None:
        return self.buffer.volatility(lookback_sec)

    def momentum(self, window_sec: int = MOMENTUM_WINDOW_SEC) -> float | None:
        return self.buffer.momentum(window_sec)

    def is_stale(self, now: datetime | None = None) -> bool:
        return self.buffer.is_stale(now)
