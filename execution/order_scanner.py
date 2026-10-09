"""Kalshi WebSocket order-book scanner for the ``KXBTC15M`` series.

Maintains local order books from the ``orderbook_delta`` channel, prices every update with the
digital-option model, and emits ``ScanCandidate``s whose edge exceeds
``StrategyParameters.min_gross_edge``.

Kalshi books only carry bids: a YES ask at ``p`` is a NO bid at ``100 - p`` and vice versa.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import itertools
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from core.schemas import Btc15mMarketState, FairValueOutput, MarketQuote, StrategyParameters
from execution.kalshi_client import KalshiClient, KalshiMarket

logger = logging.getLogger(__name__)

SERIES_TICKER = "KXBTC15M"
ORDERBOOK_CHANNEL = "orderbook_delta"

FairValueFn = Callable[[Btc15mMarketState, MarketQuote], FairValueOutput]


@dataclass(frozen=True)
class SpotContext:
    """Latest BTC spot features needed to build a ``Btc15mMarketState``."""

    spot_price: float
    historical_volatility: float
    momentum_1m: float


SpotProvider = Callable[[], SpotContext | None]
CandidateCallback = Callable[["ScanCandidate"], Awaitable[None] | None]


class WebSocketLike(Protocol):
    async def send(self, message: str) -> None: ...
    async def recv(self) -> str | bytes: ...


WebSocketConnect = Callable[[str, dict[str, str]], contextlib.AbstractAsyncContextManager[WebSocketLike]]


class OrderBookOutOfSync(Exception):
    """A sequence gap was detected; the local books must be rebuilt from a fresh snapshot."""


class KalshiWebSocketError(Exception):
    pass


class ScanCandidate(BaseModel):
    ticker: str
    side: Literal["yes", "no"]
    ask_cents: int
    model_probability: float  # probability that ``side`` pays out
    edge: float  # model_probability - ask_cents / 100
    fair_value: FairValueOutput
    quote: MarketQuote
    market_state: Btc15mMarketState
    detected_at: datetime


def load_default_fair_value_fn() -> FairValueFn:
    """Resolve ``pricing.btc_digital_option.compute_fair_value(state, quote) -> FairValueOutput``."""
    try:
        module = importlib.import_module("pricing.btc_digital_option")
    except ImportError as exc:
        raise ImportError(
            "pricing.btc_digital_option is not available; pass fair_value_fn to KalshiOrderBookScanner"
        ) from exc
    fn = getattr(module, "compute_fair_value", None)
    if not callable(fn):
        raise ImportError("pricing.btc_digital_option does not expose compute_fair_value(state, quote)")
    return fn  # type: ignore[no-any-return]


_QTY_EPSILON = 1e-6  # contract counts are fixed-point (e.g. "4.85")


def _dollars_to_cents(price: str | float) -> int:
    """Bucket a bid price into whole cents, rounding down (Kalshi quotes sub-penny levels such as
    "0.0010"). Flooring bids is conservative: the implied opposite-side ask, 100 - bid, rounds up."""
    return int(Decimal(str(price)) * 100)


def _levels(msg: dict[str, Any], side: str) -> list[tuple[int, float]]:
    if msg.get(side) is not None:
        return [(int(price), float(qty)) for price, qty in msg[side]]
    dollars = msg.get(f"{side}_dollars_fp") or msg.get(f"{side}_dollars") or []
    return [(_dollars_to_cents(price), float(qty)) for price, qty in dollars]


def _aggregate(levels: list[tuple[int, float]]) -> dict[int, float]:
    book: dict[int, float] = {}
    for price, qty in levels:
        book[price] = book.get(price, 0.0) + qty
    return {p: q for p, q in book.items() if q > _QTY_EPSILON}


class OrderBook:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.yes_bids: dict[int, float] = {}
        self.no_bids: dict[int, float] = {}
        self.updated_at: datetime | None = None

    def apply_snapshot(self, msg: dict[str, Any], ts: datetime) -> None:
        self.yes_bids = _aggregate(_levels(msg, "yes"))
        self.no_bids = _aggregate(_levels(msg, "no"))
        self.updated_at = ts

    def apply_delta(self, msg: dict[str, Any], ts: datetime) -> None:
        if msg.get("price") is not None:
            price = int(msg["price"])
        else:
            price = _dollars_to_cents(msg["price_dollars"])
        delta = float(msg["delta"] if msg.get("delta") is not None else msg["delta_fp"])
        book = self.yes_bids if msg["side"] == "yes" else self.no_bids
        qty = book.get(price, 0.0) + delta
        if qty > _QTY_EPSILON:
            book[price] = qty
        else:
            book.pop(price, None)
        self.updated_at = ts

    @property
    def best_yes_bid(self) -> int | None:
        return max(self.yes_bids) if self.yes_bids else None

    @property
    def best_no_bid(self) -> int | None:
        return max(self.no_bids) if self.no_bids else None

    @property
    def best_yes_ask(self) -> int | None:
        best_no = self.best_no_bid
        return None if best_no is None else 100 - best_no

    @property
    def best_no_ask(self) -> int | None:
        best_yes = self.best_yes_bid
        return None if best_yes is None else 100 - best_yes

    @property
    def is_crossed(self) -> bool:
        """A YES bid at or above the YES ask cannot rest on Kalshi; it means the local book is wrong."""
        bid, ask = self.best_yes_bid, self.best_yes_ask
        return bid is not None and ask is not None and bid >= ask

    @property
    def imbalance(self) -> float:
        """Top-of-book YES bid vs ask size imbalance in [-1, 1]."""
        bid_qty = self.yes_bids[max(self.yes_bids)] if self.yes_bids else 0
        ask_qty = self.no_bids[max(self.no_bids)] if self.no_bids else 0
        total = bid_qty + ask_qty
        return 0.0 if total == 0 else (bid_qty - ask_qty) / total


@contextlib.asynccontextmanager
async def _default_connect(url: str, headers: dict[str, str]) -> AsyncIterator[WebSocketLike]:
    from websockets.asyncio.client import connect

    async with connect(url, additional_headers=headers, ping_interval=10) as ws:
        yield ws


class KalshiOrderBookScanner:
    def __init__(
        self,
        client: KalshiClient,
        spot_provider: SpotProvider,
        fair_value_fn: FairValueFn | None = None,
        params: StrategyParameters | None = None,
        *,
        series_ticker: str = SERIES_TICKER,
        connect: WebSocketConnect | None = None,
        on_candidate: CandidateCallback | None = None,
        queue_maxsize: int = 1000,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
        idle_poll_seconds: float = 15.0,
        reprice_interval_seconds: float = 1.0,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._client = client
        self._spot_provider = spot_provider
        self._fair_value_fn = fair_value_fn or load_default_fair_value_fn()
        self.params = params or StrategyParameters()
        self.series_ticker = series_ticker
        self._connect = connect or _default_connect
        self._on_candidate = on_candidate
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay
        self._idle_poll_seconds = idle_poll_seconds
        self._reprice_interval = reprice_interval_seconds
        self._clock = clock

        self.candidates: asyncio.Queue[ScanCandidate] = asyncio.Queue(maxsize=queue_maxsize)
        self.markets: dict[str, KalshiMarket] = {}
        self.books: dict[str, OrderBook] = {}
        self._last_seq: dict[int, int] = {}
        self._last_emitted: dict[tuple[str, str], tuple[int, int]] = {}
        self._last_priced: dict[str, tuple[tuple[int | None, int | None], datetime]] = {}
        self._msg_ids = itertools.count(1)
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def refresh_markets(self) -> list[KalshiMarket]:
        now = self._clock()
        markets = [
            m for m in await self._client.get_markets(series_ticker=self.series_ticker) if m.close_time > now
        ]
        self.markets = {m.ticker: m for m in markets}
        self.books = {m.ticker: OrderBook(m.ticker) for m in markets}
        self._last_seq.clear()
        self._last_emitted.clear()
        self._last_priced.clear()
        return markets

    def subscribe_message(self) -> dict[str, Any]:
        return {
            "id": next(self._msg_ids),
            "cmd": "subscribe",
            "params": {"channels": [ORDERBOOK_CHANNEL], "market_tickers": sorted(self.markets)},
        }

    def _check_seq(self, envelope: dict[str, Any], *, snapshot: bool) -> None:
        sid, seq = envelope.get("sid"), envelope.get("seq")
        if sid is None or seq is None:
            return
        last = self._last_seq.get(sid)
        if not snapshot and last is not None and seq != last + 1:
            raise OrderBookOutOfSync(f"sid {sid}: expected seq {last + 1}, got {seq}")
        self._last_seq[sid] = seq

    def handle_message(self, raw: str | bytes | dict[str, Any]) -> list[ScanCandidate]:
        """Apply one WebSocket message to the local books; return any new edge candidates."""
        envelope = raw if isinstance(raw, dict) else json.loads(raw)
        msg_type = envelope.get("type")
        msg = envelope.get("msg") or {}

        if msg_type == "error":
            raise KalshiWebSocketError(f"Kalshi WebSocket error: {msg}")
        if msg_type not in ("orderbook_snapshot", "orderbook_delta"):
            logger.debug("ignoring %s message", msg_type)
            return []

        is_snapshot = msg_type == "orderbook_snapshot"
        self._check_seq(envelope, snapshot=is_snapshot)
        book = self.books.get(msg.get("market_ticker", ""))
        if book is None:
            return []
        now = self._clock()
        if is_snapshot:
            book.apply_snapshot(msg, now)
        else:
            book.apply_delta(msg, now)
        # Busy books send ~1,000 deltas/s, almost all below the top of book. Re-price only when the
        # touch moves, or periodically so spot moves are still picked up; pricing every delta starves
        # the event loop until the WebSocket keepalive times out.
        top = (book.best_yes_bid, book.best_yes_ask)
        last = self._last_priced.get(book.ticker)
        if last is not None and last[0] == top and (now - last[1]).total_seconds() < self._reprice_interval:
            return []
        self._last_priced[book.ticker] = (top, now)
        return self._new_candidates(book.ticker)

    def evaluate(self, ticker: str) -> list[ScanCandidate]:
        """Price ``ticker`` against its current book; return sides whose edge exceeds the threshold."""
        market, book = self.markets.get(ticker), self.books.get(ticker)
        spot = self._spot_provider()
        if market is None or book is None or spot is None:
            return []
        yes_bid, yes_ask = book.best_yes_bid, book.best_yes_ask
        if yes_bid is None or yes_ask is None:
            return []
        if book.is_crossed:
            logger.warning("%s book is crossed (bid %dc >= ask %dc); not pricing it", ticker, yes_bid, yes_ask)
            return []

        now = self._clock()
        seconds_remaining = int((market.close_time - now).total_seconds())
        if seconds_remaining <= 0:
            return []

        quote = MarketQuote(
            ticker=ticker,
            strike_price=market.strike_price,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            volume=market.volume,
            updated_at=book.updated_at or now,
        )
        state = Btc15mMarketState(
            ticker=ticker,
            strike_price=market.strike_price,
            expiration_time=market.close_time,
            seconds_remaining=seconds_remaining,
            current_spot_price=spot.spot_price,
            historical_volatility=spot.historical_volatility,
            momentum_1m=spot.momentum_1m,
            order_book_imbalance=book.imbalance,
        )
        fair_value = self._fair_value_fn(state, quote)

        no_ask = 100 - yes_bid
        sides: list[tuple[Literal["yes", "no"], int, float]] = [
            ("yes", yes_ask, fair_value.model_probability),
            ("no", no_ask, 1.0 - fair_value.model_probability),
        ]
        candidates = []
        for side, ask, prob in sides:
            edge = prob - ask / 100
            if edge > self.params.min_gross_edge:
                candidates.append(
                    ScanCandidate(
                        ticker=ticker,
                        side=side,
                        ask_cents=ask,
                        model_probability=prob,
                        edge=edge,
                        fair_value=fair_value,
                        quote=quote,
                        market_state=state,
                        detected_at=now,
                    )
                )
        return candidates

    def _new_candidates(self, ticker: str) -> list[ScanCandidate]:
        """Like ``evaluate`` but suppresses repeats with an unchanged ask and fair value."""
        fresh = []
        live_sides: set[tuple[str, str]] = set()
        for candidate in self.evaluate(ticker):
            key: tuple[str, str] = (ticker, candidate.side)
            live_sides.add(key)
            signature = (candidate.ask_cents, candidate.fair_value.fair_cents)
            if self._last_emitted.get(key) != signature:
                self._last_emitted[key] = signature
                fresh.append(candidate)
        for key in [k for k in self._last_emitted if k[0] == ticker and k not in live_sides]:
            del self._last_emitted[key]
        return fresh

    async def _emit(self, candidate: ScanCandidate) -> None:
        if self.candidates.full():
            dropped = self.candidates.get_nowait()
            logger.warning("candidate queue full; dropping stale %s %s", dropped.ticker, dropped.side)
        self.candidates.put_nowait(candidate)
        if self._on_candidate is not None:
            result = self._on_candidate(candidate)
            if inspect.isawaitable(result):
                await result

    async def run_session(self) -> None:
        """One connection: discover open markets, subscribe, and stream until the earliest
        market closes (so the next 15-minute window gets picked up) or ``stop()``. A dropped
        socket raises ``websockets.ConnectionClosed`` so ``run()`` reconnects with backoff."""
        markets = await self.refresh_markets()
        if not markets:
            logger.info("no open %s markets; retrying in %.0fs", self.series_ticker, self._idle_poll_seconds)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), self._idle_poll_seconds)
            return

        rollover_at = min(m.close_time for m in markets)
        async with self._connect(self._client.ws_url, self._client.ws_auth_headers()) as ws:
            await ws.send(json.dumps(self.subscribe_message()))
            logger.info("subscribed to %d %s order books", len(markets), self.series_ticker)
            while not self._stop.is_set():
                timeout = (rollover_at - self._clock()).total_seconds()
                if timeout <= 0:
                    return
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout)
                except TimeoutError:
                    return
                for candidate in self.handle_message(raw):
                    await self._emit(candidate)

    async def run(self) -> None:
        """Run sessions until ``stop()``, reconnecting with exponential backoff on failure."""
        delay = self._reconnect_delay
        while not self._stop.is_set():
            try:
                await self.run_session()
                delay = self._reconnect_delay
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("scanner session failed; reconnecting in %.1fs", delay)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), delay)
                delay = min(delay * 2, self._max_reconnect_delay)
