"""Record-only order books from the exchanges behind the CF Benchmarks BTC index.

KXBTC15M strikes and settlements are 60-second averages of the CF Benchmarks Bitcoin Real Time
Index (BRTI), which is computed from the consolidated order books of its constituent exchanges.
The trading model sees Coinbase only (shifted by a measured gap); this module keeps a live order
book per constituent we can reach so the recorder can store index proxies next to every snapshot.

Each venue streams its order book. (The first version used ticker channels: Kraken's fires on
trades only and Bullish's and Crypto.com's are throttled, so the composite lagged Coinbase by
1-3 seconds and priced the final minute worse than Coinbase alone.) From the books we derive:

    <venue>            best bid/ask mid
    <venue>_depth      mid of the volume-weighted prices to buy and to sell DEPTH_BTC, which looks
                       past the dust orders that pin the top of thinner books (Bitstamp)
    composite          median of the venue mids
    depth_composite    median of the venue depth mids
    brti_estimate      the BRTI calculation (mid price-volume curve over the consolidated book,
                       exponentially weighted up to the utilized depth) on the depth we can see;
                       CF sees deeper books and two more venues, so this is an approximation

Nothing here trades. Each exchange runs on its own reconnecting WebSocket with a fresh book per
connection; a dead or stale feed only drops that venue from the samples.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import numpy as np
import websockets

from core.constants import WS_RECONNECT_INITIAL_SEC, WS_RECONNECT_MAX_SEC

logger = logging.getLogger(__name__)

QUOTE_MAX_AGE_SEC = 10.0  # a venue whose book has not changed for this long is left out
DEPTH_BTC = 1.0  # size walked on each side for the depth mid
BOOK_WINDOW = 0.01  # levels further than 1% from the best price are ignored by the estimates
MAX_VENUE_DEVIATION = 0.01  # a venue this far from the median of the others is treated as broken
KRAKEN_DEPTH = 100
RTI_SPREAD_LIMIT = 0.005  # utilized depth ends where the mid spread-volume curve reaches 0.5%
RTI_LAMBDA_FRACTION = 0.3  # exponential weights with lambda = 1 / (0.3 * utilized depth)
RTI_SIZE_CAP_SD = 5.0  # order sizes are capped at the trimmed mean + 5 sd (CF's dynamic order-size cap)
RTI_GRID_POINTS = 200

Side = Literal["bid", "ask"]
Level = tuple[float, float]  # (price, size in BTC)


class BookGap(Exception):
    """An incremental feed skipped a sequence number; the book must be rebuilt (reconnect)."""


class Book:
    """One venue's order book: price -> size per side."""

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def replace(self, bids: Iterable[Level], asks: Iterable[Level]) -> None:
        self.bids = {p: q for p, q in bids if q > 0}
        self.asks = {p: q for p, q in asks if q > 0}

    def set(self, side: Side, price: float, size: float) -> None:
        levels = self.bids if side == "bid" else self.asks
        if size > 0:
            levels[price] = size
        else:
            levels.pop(price, None)

    def truncate(self, depth: int) -> None:
        """Keep the best `depth` levels per side (feeds that only maintain a fixed depth)."""
        if len(self.bids) > depth:
            self.bids = dict(sorted(self.bids.items(), reverse=True)[:depth])
        if len(self.asks) > depth:
            self.asks = dict(sorted(self.asks.items())[:depth])

    def best(self) -> Optional[tuple[float, float]]:
        if not self.bids or not self.asks:
            return None
        bid, ask = max(self.bids), min(self.asks)
        return (bid, ask) if bid < ask else None  # a crossed book is a broken book

    def mid(self) -> Optional[float]:
        best = self.best()
        return None if best is None else (best[0] + best[1]) / 2

    def levels(self, side: Side, window: float = BOOK_WINDOW) -> list[Level]:
        """Levels within `window` of the best price on that side, best first."""
        best = self.best()
        if best is None:
            return []
        if side == "bid":
            floor = best[0] * (1 - window)
            return sorted(((p, q) for p, q in self.bids.items() if p >= floor), reverse=True)
        cap = best[1] * (1 + window)
        return sorted((p, q) for p, q in self.asks.items() if p <= cap)

    def depth_mid(self, depth: float = DEPTH_BTC) -> Optional[float]:
        bid, ask = _vwap(self.levels("bid"), depth), _vwap(self.levels("ask"), depth)
        return None if bid is None or ask is None else (bid + ask) / 2


def _vwap(levels: list[Level], depth: float) -> Optional[float]:
    filled = cost = 0.0
    for price, size in levels:
        take = min(size, depth - filled)
        filled += take
        cost += take * price
        if filled >= depth - 1e-12:
            return cost / depth
    return None  # not enough size on this side


def rti_estimate(books: Iterable[Book]) -> Optional[float]:
    """The BRTI calculation applied to the consolidated `books`."""
    bids = sorted((lv for b in books for lv in b.levels("bid")), reverse=True)
    asks = sorted(lv for b in books for lv in b.levels("ask"))
    if not bids or not asks:
        return None
    sizes = np.array([q for _, q in bids] + [q for _, q in asks])
    trim = len(sizes) // 100
    core = np.sort(sizes)[trim : len(sizes) - trim] if trim else sizes
    cap = core.mean() + RTI_SIZE_CAP_SD * core.std()
    bid_px, bid_cum = np.array([p for p, _ in bids]), np.cumsum(np.minimum([q for _, q in bids], cap))
    ask_px, ask_cum = np.array([p for p, _ in asks]), np.cumsum(np.minimum([q for _, q in asks], cap))
    volumes = np.linspace(0, min(bid_cum[-1], ask_cum[-1]), RTI_GRID_POINTS + 1)[1:]
    # Marginal price of the v-th unit on each side, and the mid / relative spread at each volume.
    buy = ask_px[np.minimum(np.searchsorted(ask_cum, volumes), len(ask_px) - 1)]
    sell = bid_px[np.minimum(np.searchsorted(bid_cum, volumes), len(bid_px) - 1)]
    mid = (buy + sell) / 2
    within = buy / mid - 1 <= RTI_SPREAD_LIMIT
    n = len(volumes) if within.all() else int(np.argmin(within))
    if n == 0:
        return float((bids[0][0] + asks[0][0]) / 2)
    lam = 1 / (RTI_LAMBDA_FRACTION * volumes[n - 1])
    weights = lam * np.exp(-lam * volumes[:n])
    return float(np.dot(weights, mid[:n]) / weights.sum())


# ---------------------------------------------------------------------------
# Per-exchange handlers: apply one message to the book, return True if it touched the book.
# Handler factories are called once per connection (some keep sequence state).
# ---------------------------------------------------------------------------

Handler = Callable[[dict[str, Any], Book], bool]


def _pairs(rows: Iterable[Any]) -> list[Level]:
    """[[price, size, ...], ...] with string or number fields (Crypto.com adds an order count)."""
    return [(float(r[0]), float(r[1])) for r in rows]


def _flat_pairs(flat: list[Any]) -> list[Level]:
    """[price, size, price, size, ...] (Bullish)."""
    return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat) - 1, 2)]


def handle_coinbase(msg: dict[str, Any], book: Book) -> bool:
    if msg.get("product_id") != "BTC-USD":
        return False
    if msg.get("type") == "snapshot":
        book.replace(_pairs(msg.get("bids") or []), _pairs(msg.get("asks") or []))
        return True
    if msg.get("type") == "l2update":
        for side, price, size in msg.get("changes") or []:
            book.set("bid" if side == "buy" else "ask", float(price), float(size))
        return True
    return False


def handle_kraken(msg: dict[str, Any], book: Book) -> bool:
    if msg.get("channel") != "book" or not msg.get("data"):
        return False
    row = msg["data"][0]
    bids = [(float(lv["price"]), float(lv["qty"])) for lv in row.get("bids") or []]
    asks = [(float(lv["price"]), float(lv["qty"])) for lv in row.get("asks") or []]
    if msg.get("type") == "snapshot":
        book.replace(bids, asks)
    else:
        for price, size in bids:
            book.set("bid", price, size)
        for price, size in asks:
            book.set("ask", price, size)
        book.truncate(KRAKEN_DEPTH)  # Kraken expects clients to drop levels pushed out of the depth
    return True


def handle_bitstamp(msg: dict[str, Any], book: Book) -> bool:
    if msg.get("event") != "data" or msg.get("channel") != "order_book_btcusd":
        return False
    data = msg.get("data") or {}
    book.replace(_pairs(data.get("bids") or []), _pairs(data.get("asks") or []))  # top 100, every ~100 ms
    return True


def handle_gemini(msg: dict[str, Any], book: Book) -> bool:
    """v2 `l2`: the first update carries the whole book, later ones the changed levels."""
    if msg.get("type") != "l2_updates" or msg.get("symbol") != "BTCUSD":
        return False
    for side, price, size in msg.get("changes") or []:
        book.set("bid" if side == "buy" else "ask", float(price), float(size))
    return True


class CryptocomHandler:
    """`book.BTC_USD.50` as a snapshot followed by deltas chained by sequence number."""

    def __init__(self) -> None:
        self.seq: Optional[int] = None

    def __call__(self, msg: dict[str, Any], book: Book) -> bool:
        result = msg.get("result") or {}
        channel = result.get("channel")
        if channel not in ("book", "book.update") or not result.get("data"):
            return False
        row = result["data"][0]
        if channel == "book":
            book.replace(_pairs(row.get("bids") or []), _pairs(row.get("asks") or []))
        else:
            if self.seq is not None and row.get("pu") != self.seq:
                raise BookGap(f"crypto.com book update {row.get('pu')} does not follow {self.seq}")
            update = row.get("update") or {}
            for price, size in _pairs(update.get("bids") or []):
                book.set("bid", price, size)
            for price, size in _pairs(update.get("asks") or []):
                book.set("ask", price, size)
        self.seq = row.get("u")
        return True


def handle_bullish(msg: dict[str, Any], book: Book) -> bool:
    if msg.get("dataType") != "V1TALevel2":
        return False
    data = msg.get("data") or {}
    if data.get("symbol") != "BTCUSD":
        return False
    bids, asks = _flat_pairs(data.get("bids") or []), _flat_pairs(data.get("asks") or [])
    if msg.get("type") == "update":
        for price, size in bids:
            book.set("bid", price, size)
        for price, size in asks:
            book.set("ask", price, size)
    else:
        book.replace(bids, asks)
    return True


@dataclass(frozen=True)
class ExchangeSpec:
    name: str
    url: str
    subscribe: Optional[dict[str, Any]]
    handler: Callable[[], Handler]  # factory: a fresh (possibly stateful) handler per connection


EXCHANGES: tuple[ExchangeSpec, ...] = (
    ExchangeSpec(
        "coinbase",
        "wss://ws-feed.exchange.coinbase.com",
        {"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["level2_batch"]},
        lambda: handle_coinbase,
    ),
    ExchangeSpec(
        "kraken",
        "wss://ws.kraken.com/v2",
        {"method": "subscribe", "params": {"channel": "book", "symbol": ["BTC/USD"], "depth": KRAKEN_DEPTH}},
        lambda: handle_kraken,
    ),
    ExchangeSpec(
        "bitstamp",
        "wss://ws.bitstamp.net",
        {"event": "bts:subscribe", "data": {"channel": "order_book_btcusd"}},
        lambda: handle_bitstamp,
    ),
    ExchangeSpec(
        "gemini",
        "wss://api.gemini.com/v2/marketdata",
        {"type": "subscribe", "subscriptions": [{"name": "l2", "symbols": ["BTCUSD"]}]},
        lambda: handle_gemini,
    ),
    ExchangeSpec(
        "cryptocom",
        "wss://stream.crypto.com/exchange/v1/market",
        {
            "id": 1,
            "method": "subscribe",
            "params": {
                "channels": ["book.BTC_USD.50"],
                "book_subscription_type": "SNAPSHOT_AND_UPDATE",
                "book_update_frequency": 10,
            },
        },
        CryptocomHandler,
    ),
    ExchangeSpec(
        "bullish",
        "wss://api.exchange.bullish.com/trading-api/v1/market-data/orderbook",
        {
            "jsonrpc": "2.0",
            "type": "command",
            "method": "subscribe",
            "params": {"topic": "l2Orderbook", "symbol": "BTCUSD"},
            "id": "1",
        },
        lambda: handle_bullish,
    ),
)
SOURCES: tuple[str, ...] = tuple(e.name for e in EXCHANGES)
SAMPLE_FIELDS: tuple[str, ...] = (
    *SOURCES,
    *(f"{s}_depth" for s in SOURCES),
    "composite",
    "depth_composite",
    "brti_estimate",
)


def composite(mids: dict[str, float]) -> Optional[float]:
    """Median of the available venue mids (robust to one bad or lagging venue)."""
    return statistics.median(mids.values()) if mids else None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ExchangeRecorder:
    """Keeps a live order book per exchange. `start()` runs one reconnecting task per venue."""

    def __init__(
        self,
        exchanges: tuple[ExchangeSpec, ...] = EXCHANGES,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.exchanges = exchanges
        self._clock = clock
        self.books: dict[str, Book] = {}
        self.updated_at: dict[str, datetime] = {}
        self.connected: dict[str, bool] = {e.name: False for e in exchanges}
        self.reconnects: dict[str, int] = {e.name: 0 for e in exchanges}
        self._tasks: list[asyncio.Task[None]] = []

    # -- state -------------------------------------------------------------

    def update(self, source: str, bid: float, ask: float, size: float = 10.0) -> None:
        """Set a venue's book to a single level per side (tests and manual quotes)."""
        book = Book()
        book.replace([(bid, size)], [(ask, size)])
        self.books[source] = book
        self.updated_at[source] = self._clock()

    def fresh_books(self, now: Optional[datetime] = None) -> dict[str, Book]:
        """Venues with a recent, uncrossed book that agrees with the others to within 1%."""
        now = now or self._clock()
        books = {
            name: book
            for name, book in self.books.items()
            if (now - self.updated_at[name]).total_seconds() <= QUOTE_MAX_AGE_SEC and book.best() is not None
        }
        if len(books) < 3:
            return books
        mids = {name: b.mid() or 0.0 for name, b in books.items()}
        center = statistics.median(mids.values())
        return {n: b for n, b in books.items() if abs(mids[n] / center - 1) <= MAX_VENUE_DEVIATION}

    def mids(self, now: Optional[datetime] = None) -> dict[str, float]:
        """Fresh best bid/ask mids by venue."""
        return {name: book.mid() or 0.0 for name, book in self.fresh_books(now).items()}

    def composite(self, now: Optional[datetime] = None) -> Optional[float]:
        return composite(self.mids(now))

    def sample(self, now: Optional[datetime] = None) -> dict[str, Any]:
        """Every index proxy at this instant, keyed like `db.models.IndexSample`."""
        books = self.fresh_books(now)
        mids = {name: book.mid() or 0.0 for name, book in books.items()}
        depth = {f"{name}_depth": m for name, b in books.items() if (m := b.depth_mid()) is not None}
        return {
            **mids,
            **depth,
            "composite": composite(mids),
            "depth_composite": composite(depth),
            "brti_estimate": rti_estimate(books.values()),
            "sources": len(mids),
        }

    def status(self, now: Optional[datetime] = None) -> list[dict[str, Any]]:
        now = now or self._clock()
        fresh = self.fresh_books(now)
        out = []
        for e in self.exchanges:
            book = self.books.get(e.name)
            updated = self.updated_at.get(e.name)
            out.append(
                {
                    "source": e.name,
                    "connected": self.connected[e.name],
                    "mid": book.mid() if book else None,
                    "depth_mid": book.depth_mid() if book else None,
                    "age_sec": (now - updated).total_seconds() if updated else None,
                    "fresh": e.name in fresh,
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
                # Full-book snapshots (Coinbase, Gemini) run to several MB.
                async with websockets.connect(spec.url, open_timeout=15, ping_interval=20, max_size=2**24) as ws:
                    if spec.subscribe is not None:
                        await ws.send(json.dumps(spec.subscribe))
                    self.connected[spec.name] = True
                    logger.info("exchange recorder connected: %s", spec.name)
                    handler, book = spec.handler(), Book()
                    self.books[spec.name] = book
                    async for raw in ws:
                        msg = json.loads(raw)
                        if not isinstance(msg, dict):
                            continue
                        if msg.get("method") == "public/heartbeat":  # crypto.com drops silent clients
                            await ws.send(json.dumps({"id": msg.get("id"), "method": "public/respond-heartbeat"}))
                            continue
                        if handler(msg, book):
                            self.updated_at[spec.name] = self._clock()
                            backoff = WS_RECONNECT_INITIAL_SEC
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # network errors, closed sockets, bad JSON, book gaps: reconnect
                logger.info("exchange recorder %s disconnected (%s); retrying in %.0fs", spec.name, exc, backoff)
            self.connected[spec.name] = False
            self.reconnects[spec.name] += 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, WS_RECONNECT_MAX_SEC)
