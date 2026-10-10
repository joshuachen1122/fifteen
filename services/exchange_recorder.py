"""Record-only top-of-book feeds from the exchanges behind the CF Benchmarks BTC index.

KXBTC15M strikes and settlements are 60-second averages of the CF Benchmarks Bitcoin Real Time
Index, which aggregates order books across several exchanges. The trading model sees Coinbase
only (shifted by a measured gap). This module keeps the latest best bid/ask from each index
constituent we can reach, so the recorder can store their mids and their median ("composite")
next to every snapshot. That gives future models a much closer proxy for the settlement index
than Coinbase alone, and lets us fit the index from data instead of a single median gap.

Nothing here trades. Each exchange runs on its own reconnecting WebSocket; a dead feed only
drops that source from the composite.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import websockets

from core.constants import WS_RECONNECT_INITIAL_SEC, WS_RECONNECT_MAX_SEC

logger = logging.getLogger(__name__)

QUOTE_MAX_AGE_SEC = 10.0  # older quotes are left out of samples and the composite


@dataclass(frozen=True)
class TopOfBook:
    bid: float
    ask: float
    received_at: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


def _quote(bid: Any, ask: Any) -> Optional[tuple[float, float]]:
    try:
        b, a = float(bid), float(ask)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(b) and math.isfinite(a)) or b <= 0 or a <= 0 or a < b:
        return None
    return b, a


# ---------------------------------------------------------------------------
# Per-exchange subscriptions and parsers. A parser returns (bid, ask), or None for messages
# that carry no quote. Parsers may keep state (Gemini sends one side per event).
# ---------------------------------------------------------------------------


def parse_coinbase(msg: dict[str, Any]) -> Optional[tuple[float, float]]:
    if msg.get("type") != "ticker" or msg.get("product_id") != "BTC-USD":
        return None
    return _quote(msg.get("best_bid"), msg.get("best_ask"))


def parse_kraken(msg: dict[str, Any]) -> Optional[tuple[float, float]]:
    if msg.get("channel") != "ticker" or not msg.get("data"):
        return None
    row = msg["data"][0]
    return _quote(row.get("bid"), row.get("ask"))


def parse_bitstamp(msg: dict[str, Any]) -> Optional[tuple[float, float]]:
    if msg.get("event") != "data" or msg.get("channel") != "order_book_btcusd":
        return None
    data = msg.get("data") or {}
    bids, asks = data.get("bids") or [], data.get("asks") or []
    if not bids or not asks:
        return None
    return _quote(bids[0][0], asks[0][0])


def parse_cryptocom(msg: dict[str, Any]) -> Optional[tuple[float, float]]:
    result = msg.get("result") or {}
    if result.get("channel") != "ticker" or not result.get("data"):
        return None
    row = result["data"][0]
    return _quote(row.get("b"), row.get("k"))


def parse_bullish(msg: dict[str, Any]) -> Optional[tuple[float, float]]:
    if msg.get("dataType") != "V1TATickerResponse":
        return None
    data = msg.get("data") or {}
    return _quote(data.get("bestBid"), data.get("bestAsk"))


class GeminiParser:
    """Gemini's top-of-book stream sends bid and ask changes as separate events."""

    def __init__(self) -> None:
        self.bid: Optional[float] = None
        self.ask: Optional[float] = None

    def __call__(self, msg: dict[str, Any]) -> Optional[tuple[float, float]]:
        if msg.get("type") != "update":
            return None
        for event in msg.get("events") or []:
            if event.get("type") != "change":
                continue
            try:
                price = float(event["price"])
            except (KeyError, TypeError, ValueError):
                continue
            if event.get("side") == "bid":
                self.bid = price
            elif event.get("side") == "ask":
                self.ask = price
        return _quote(self.bid, self.ask) if self.bid is not None and self.ask is not None else None


Parser = Callable[[dict[str, Any]], Optional[tuple[float, float]]]


@dataclass(frozen=True)
class ExchangeSpec:
    name: str
    url: str
    subscribe: Optional[dict[str, Any]]
    parser: Callable[[], Parser]  # factory: a fresh (possibly stateful) parser per connection


EXCHANGES: tuple[ExchangeSpec, ...] = (
    ExchangeSpec(
        "coinbase",
        "wss://ws-feed.exchange.coinbase.com",
        {"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]},
        lambda: parse_coinbase,
    ),
    ExchangeSpec(
        "kraken",
        "wss://ws.kraken.com/v2",
        {"method": "subscribe", "params": {"channel": "ticker", "symbol": ["BTC/USD"]}},
        lambda: parse_kraken,
    ),
    ExchangeSpec(
        "bitstamp",
        "wss://ws.bitstamp.net",
        {"event": "bts:subscribe", "data": {"channel": "order_book_btcusd"}},
        lambda: parse_bitstamp,
    ),
    ExchangeSpec(
        "gemini",
        "wss://api.gemini.com/v1/marketdata/BTCUSD?top_of_book=true&trades=false&heartbeat=true",
        None,
        GeminiParser,
    ),
    ExchangeSpec(
        "cryptocom",
        "wss://stream.crypto.com/exchange/v1/market",
        {"id": 1, "method": "subscribe", "params": {"channels": ["ticker.BTC_USD"]}},
        lambda: parse_cryptocom,
    ),
    ExchangeSpec(
        "bullish",
        "wss://api.exchange.bullish.com/trading-api/v1/market-data/tick/BTCUSD",
        None,
        lambda: parse_bullish,
    ),
)
SOURCES: tuple[str, ...] = tuple(e.name for e in EXCHANGES)


def composite(mids: dict[str, float]) -> Optional[float]:
    """Median of the available exchange mids (robust to one bad or lagging venue)."""
    return statistics.median(mids.values()) if mids else None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ExchangeRecorder:
    """Keeps the latest top of book per exchange. `start()` runs one reconnecting task per venue."""

    def __init__(
        self,
        exchanges: tuple[ExchangeSpec, ...] = EXCHANGES,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.exchanges = exchanges
        self._clock = clock
        self.books: dict[str, TopOfBook] = {}
        self.connected: dict[str, bool] = {e.name: False for e in exchanges}
        self.reconnects: dict[str, int] = {e.name: 0 for e in exchanges}
        self._tasks: list[asyncio.Task[None]] = []

    # -- state -------------------------------------------------------------

    def update(self, source: str, bid: float, ask: float) -> None:
        self.books[source] = TopOfBook(bid=bid, ask=ask, received_at=self._clock())

    def mids(self, now: Optional[datetime] = None) -> dict[str, float]:
        """Fresh mids by source."""
        now = now or self._clock()
        return {
            name: book.mid
            for name, book in self.books.items()
            if (now - book.received_at).total_seconds() <= QUOTE_MAX_AGE_SEC
        }

    def composite(self, now: Optional[datetime] = None) -> Optional[float]:
        return composite(self.mids(now))

    def status(self, now: Optional[datetime] = None) -> list[dict[str, Any]]:
        now = now or self._clock()
        out = []
        for e in self.exchanges:
            book = self.books.get(e.name)
            age = (now - book.received_at).total_seconds() if book else None
            out.append(
                {
                    "source": e.name,
                    "connected": self.connected[e.name],
                    "mid": book.mid if book else None,
                    "age_sec": age,
                    "fresh": age is not None and age <= QUOTE_MAX_AGE_SEC,
                    "reconnects": self.reconnects[e.name],
                }
            )
        return out

    # -- connections ---------------------------------------------------------

    def start(self) -> asyncio.Task[None]:
        async def run_all() -> None:
            self._tasks = [asyncio.create_task(self._run(e), name=f"exchange-{e.name}") for e in self.exchanges]
            await asyncio.gather(*self._tasks, return_exceptions=True)

        return asyncio.create_task(run_all(), name="exchange-recorder")

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _run(self, spec: ExchangeSpec) -> None:
        backoff = WS_RECONNECT_INITIAL_SEC
        while True:
            try:
                async with websockets.connect(spec.url, open_timeout=15, ping_interval=20, max_size=2**22) as ws:
                    if spec.subscribe is not None:
                        await ws.send(json.dumps(spec.subscribe))
                    self.connected[spec.name] = True
                    logger.info("exchange recorder connected: %s", spec.name)
                    parser = spec.parser()
                    async for raw in ws:
                        msg = json.loads(raw)
                        if not isinstance(msg, dict):
                            continue
                        if msg.get("method") == "public/heartbeat":  # crypto.com drops silent clients
                            await ws.send(json.dumps({"id": msg.get("id"), "method": "public/respond-heartbeat"}))
                            continue
                        quote = parser(msg)
                        if quote is not None:
                            self.update(spec.name, *quote)
                            backoff = WS_RECONNECT_INITIAL_SEC
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # network errors, closed sockets, bad JSON: reconnect
                logger.info("exchange recorder %s disconnected (%s); retrying in %.0fs", spec.name, exc, backoff)
            self.connected[spec.name] = False
            self.reconnects[spec.name] += 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, WS_RECONNECT_MAX_SEC)
