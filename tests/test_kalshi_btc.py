from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import sys
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from websockets.exceptions import ConnectionClosedOK

from core.schemas import Btc15mMarketState, FairValueOutput, MarketQuote, OrderRequest, StrategyParameters
from execution.kalshi_client import (
    DEMO_BASE_URL,
    PROD_BASE_URL,
    KalshiAPIError,
    KalshiClient,
    KalshiMarket,
    KalshiRequestSigner,
    PaperTradingError,
)
from execution.order_scanner import (
    KalshiOrderBookScanner,
    KalshiWebSocketError,
    OrderBook,
    OrderBookOutOfSync,
    ScanCandidate,
    SpotContext,
)
from execution.rate_limiter import TokenBucketRateLimiter

NOW = datetime(2026, 10, 9, 14, 0, 0, tzinfo=timezone.utc)
CLOSE = NOW + timedelta(minutes=10)
TICKER = "KXBTC15M-26OCT09-B82500"


# --------------------------------------------------------------------------- fixtures


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(scope="module")
def private_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def signer(private_key: rsa.RSAPrivateKey) -> KalshiRequestSigner:
    return KalshiRequestSigner("test-key-id", private_key)


def verify_signature(key: rsa.RSAPrivateKey, signature_b64: str, message: str) -> None:
    key.public_key().verify(
        base64.b64decode(signature_b64),
        message.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


class Recorder:
    """httpx MockTransport handler that records requests and replays canned responses."""

    def __init__(self, routes: dict[tuple[str, str], Any]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self.routes.get((request.method, request.url.path))
        if response is None:
            return httpx.Response(404, json={"error": {"code": "not_found"}})
        if callable(response):
            response = response(request)
        if isinstance(response, httpx.Response):
            return response
        return httpx.Response(200, json=response)


def unlimited() -> TokenBucketRateLimiter:
    return TokenBucketRateLimiter(rate=1e9)


def make_client(handler: Any, *, signer: KalshiRequestSigner | None, trading_enabled: bool) -> KalshiClient:
    return KalshiClient(
        base_url=DEMO_BASE_URL,
        signer=signer,
        trading_enabled=trading_enabled,
        rate_limiter=unlimited(),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def no_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected network call: {request.method} {request.url}")


def raw_market(ticker: str = TICKER, close: datetime = CLOSE, **extra: Any) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "event_ticker": "KXBTC15M-26OCT09",
        "status": "active",
        "floor_strike": 82500,
        "close_time": close.isoformat().replace("+00:00", "Z"),
        "yes_bid": 40,
        "yes_ask": 45,
        "volume": 1234,
        **extra,
    }


# --------------------------------------------------------------------------- rate limiter


async def test_rate_limiter_allows_burst_up_to_capacity() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(rate=10, clock=clock, sleep=clock.sleep)
    for _ in range(10):
        await limiter.acquire()
    assert clock.sleeps == []
    assert limiter.available_tokens == pytest.approx(0.0)


async def test_rate_limiter_waits_for_refill_when_exhausted() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(rate=10, clock=clock, sleep=clock.sleep)
    for _ in range(10):
        await limiter.acquire()
    await limiter.acquire()
    assert clock.now == pytest.approx(0.1)


async def test_rate_limiter_sustained_throughput_matches_rate() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(rate=10, clock=clock, sleep=clock.sleep)
    await asyncio.gather(*(limiter.acquire() for _ in range(30)))
    # 10 burst tokens, then 20 more at 10/s.
    assert clock.now == pytest.approx(2.0)


async def test_rate_limiter_refill_is_capped_at_capacity() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(rate=5, capacity=3, clock=clock, sleep=clock.sleep)
    await limiter.acquire(3)
    clock.now += 100
    assert limiter.available_tokens == pytest.approx(3.0)


async def test_rate_limiter_try_acquire_and_context_manager() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(rate=2, clock=clock, sleep=clock.sleep)
    assert limiter.try_acquire()
    async with limiter:
        pass
    assert not limiter.try_acquire()
    clock.now += 0.5
    assert limiter.try_acquire()


def test_rate_limiter_rejects_invalid_config() -> None:
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(rate=0)
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(rate=10, capacity=0.5)
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(rate=10).try_acquire(11)


def test_rate_limiter_default_is_10_rps() -> None:
    limiter = TokenBucketRateLimiter()
    assert limiter.rate == 10
    assert limiter.capacity == 10


# --------------------------------------------------------------------------- signing


def test_signer_signs_timestamp_method_and_path_without_query(
    signer: KalshiRequestSigner, private_key: rsa.RSAPrivateKey
) -> None:
    headers = signer.headers("get", "/trade-api/v2/markets?series_ticker=KXBTC15M", timestamp_ms=1700000000000)
    assert headers["KALSHI-ACCESS-KEY"] == "test-key-id"
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == "1700000000000"
    verify_signature(private_key, headers["KALSHI-ACCESS-SIGNATURE"], "1700000000000GET/trade-api/v2/markets")


def test_signer_loads_from_pem(private_key: rsa.RSAPrivateKey) -> None:
    pem = private_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    loaded = KalshiRequestSigner.from_pem("kid", pem)
    headers = loaded.headers("POST", "/trade-api/v2/portfolio/orders", timestamp_ms=1)
    verify_signature(private_key, headers["KALSHI-ACCESS-SIGNATURE"], "1POST/trade-api/v2/portfolio/orders")


# --------------------------------------------------------------------------- REST client (live mode, mocked)


async def test_live_get_balance_sends_signed_request(
    signer: KalshiRequestSigner, private_key: rsa.RSAPrivateKey
) -> None:
    recorder = Recorder({("GET", "/trade-api/v2/portfolio/balance"): {"balance": 52_300}})
    async with make_client(recorder, signer=signer, trading_enabled=True) as client:
        assert await client.get_balance() == 52_300

    request = recorder.requests[0]
    ts = request.headers["KALSHI-ACCESS-TIMESTAMP"]
    verify_signature(private_key, request.headers["KALSHI-ACCESS-SIGNATURE"], f"{ts}GET/trade-api/v2/portfolio/balance")


@pytest.mark.parametrize(("side", "book_side", "yes_price"), [("yes", "bid", "0.4200"), ("no", "ask", "0.5800")])
async def test_live_submit_order_posts_v2_immediate_or_cancel(
    signer: KalshiRequestSigner, side: str, book_side: str, yes_price: str
) -> None:
    recorder = Recorder(
        {
            ("POST", "/trade-api/v2/portfolio/events/orders"): lambda request: {
                "order_id": "ord-1",
                "client_order_id": json.loads(request.content)["client_order_id"],
                "fill_count": "0.00",
                "remaining_count": "0.00",
                "ts_ms": 1,
            }
        }
    )
    order = OrderRequest(ticker=TICKER, side=side, count=4, max_cost_cents=170, kelly_fraction=0.1, reasoning="t")
    async with make_client(recorder, signer=signer, trading_enabled=True) as client:
        result = await client.submit_order(order, client_order_id="cid-1", exchange_index=2)

    sent = json.loads(recorder.requests[0].content)
    # Order V2 quotes the YES leg: buying NO at 42c (170 // 4) is an ask on YES at 58c.
    assert sent == {
        "ticker": TICKER,
        "client_order_id": "cid-1",
        "side": book_side,
        "count": "4.00",
        "price": yes_price,
        "time_in_force": "immediate_or_cancel",
        "self_trade_prevention_type": "taker_at_cross",
        "exchange_index": 2,
    }
    assert (result.order_id, result.price_cents, result.filled_count, result.status) == ("ord-1", 42, 0, "canceled")
    assert not result.paper and result.avg_fill_price_cents is None
    assert len(recorder.requests) == 1  # no fills lookup when nothing filled


async def test_live_fill_price_and_fee_come_from_the_fills_ledger(signer: KalshiRequestSigner) -> None:
    fills = {
        "fills": [
            {"order_id": "ord-2", "count_fp": "2.00", "yes_price_dollars": "0.5900", "no_price_dollars": "0.4100", "fee_cost": "0.0300"},
            {"order_id": "ord-2", "count_fp": "1.00", "yes_price_dollars": "0.5800", "no_price_dollars": "0.4200", "fee_cost": "0.0200"},
            {"order_id": "other", "count_fp": "9.00", "yes_price_dollars": "0.1000", "no_price_dollars": "0.9000", "fee_cost": "0.5000"},
        ]
    }
    recorder = Recorder(
        {
            ("POST", "/trade-api/v2/portfolio/events/orders"): {
                "order_id": "ord-2", "fill_count": "3.00", "remaining_count": "0.00",
                "average_fill_price": "0.5867", "average_fee_paid": "0.0167", "ts_ms": 1,
            },
            ("GET", "/trade-api/v2/portfolio/fills"): fills,
        }
    )
    order = OrderRequest(ticker=TICKER, side="no", count=4, max_cost_cents=4 * 42, kelly_fraction=0.1, reasoning="t")
    async with make_client(recorder, signer=signer, trading_enabled=True) as client:
        result = await client.submit_order(order)

    assert recorder.requests[1].url.params["order_id"] == "ord-2"
    assert (result.status, result.filled_count) == ("partial", 3)
    assert result.avg_fill_price_cents == pytest.approx((2 * 41 + 1 * 42) / 3)  # NO-side price
    assert result.fees_cents == 5


async def test_live_balance_can_be_scoped_to_an_exchange_shard(signer: KalshiRequestSigner) -> None:
    recorder = Recorder(
        {
            ("GET", "/trade-api/v2/portfolio/balance"): lambda request: {
                "balance": 2829 if request.url.params.get("exchange_index") == "2" else 43058
            },
            ("GET", "/trade-api/v2/series/KXBTC15M"): {"series": {"ticker": "KXBTC15M", "exchange_index": 2}},
        }
    )
    async with make_client(recorder, signer=signer, trading_enabled=True) as client:
        assert await client.get_series_exchange_index("KXBTC15M") == 2
        assert await client.get_balance(exchange_index=2) == 2829
        assert await client.get_balance() == 43058


async def test_live_cancel_order_sends_delete(signer: KalshiRequestSigner) -> None:
    recorder = Recorder(
        {
            ("DELETE", "/trade-api/v2/portfolio/orders/ord-9"): {
                "order": {
                    "order_id": "ord-9",
                    "ticker": TICKER,
                    "side": "yes",
                    "status": "canceled",
                    "yes_price": 30,
                    "initial_count": 5,
                    "remaining_count": 0,
                    "fill_count": 2,
                },
                "reduced_by": 3,
            }
        }
    )
    async with make_client(recorder, signer=signer, trading_enabled=True) as client:
        result = await client.cancel_order("ord-9")
    assert recorder.requests[0].method == "DELETE"
    assert result.status == "canceled"
    assert result.filled_count == 2


async def test_live_api_error_raises_with_status(signer: KalshiRequestSigner) -> None:
    recorder = Recorder(
        {("GET", "/trade-api/v2/portfolio/balance"): httpx.Response(429, json={"error": "too many requests"})}
    )
    async with make_client(recorder, signer=signer, trading_enabled=True) as client:
        with pytest.raises(KalshiAPIError) as excinfo:
            await client.get_balance()
    assert excinfo.value.status_code == 429


async def test_get_markets_follows_cursor_and_parses(signer: KalshiRequestSigner) -> None:
    pages = iter(
        [
            {"markets": [raw_market()], "cursor": "page2"},
            {
                "markets": [
                    raw_market("KXBTC15M-26OCT09-B83000", floor_strike=None, yes_bid=None, yes_bid_dollars="0.2100")
                ],
                "cursor": "",
            },
        ]
    )
    recorder = Recorder({("GET", "/trade-api/v2/markets"): lambda _req: next(pages)})
    async with make_client(recorder, signer=signer, trading_enabled=True) as client:
        markets = await client.get_markets()

    assert [m.ticker for m in markets] == [TICKER, "KXBTC15M-26OCT09-B83000"]
    assert markets[1].strike_price == 83000.0  # parsed from ticker suffix
    assert markets[1].yes_bid == 21  # parsed from dollars field
    assert recorder.requests[0].url.params["series_ticker"] == "KXBTC15M"
    assert recorder.requests[1].url.params["cursor"] == "page2"


async def test_requests_go_through_rate_limiter(signer: KalshiRequestSigner) -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(rate=10, capacity=1, clock=clock, sleep=clock.sleep)
    recorder = Recorder({("GET", "/trade-api/v2/portfolio/balance"): {"balance": 1}})
    client = KalshiClient(
        signer=signer,
        trading_enabled=True,
        rate_limiter=limiter,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)),
    )
    async with client:
        for _ in range(3):
            await client.get_balance()
    assert len(recorder.requests) == 3
    assert clock.now == pytest.approx(0.2)


def test_live_trading_requires_credentials() -> None:
    with pytest.raises(ValueError):
        KalshiClient(trading_enabled=True, signer=None)


@pytest.mark.parametrize(
    ("side", "count", "max_cost"),
    [("maybe", 1, 50), ("yes", 0, 50), ("yes", 1, 0), ("no", 1, 100), ("yes", 10, 5)],
)
async def test_invalid_orders_are_rejected_before_network(side: str, count: int, max_cost: int) -> None:
    order = OrderRequest(ticker=TICKER, side=side, count=count, max_cost_cents=max_cost, kelly_fraction=0, reasoning="")
    async with make_client(no_network, signer=None, trading_enabled=False) as client:
        with pytest.raises(ValueError):
            await client.submit_order(order)


# --------------------------------------------------------------------------- paper trading


async def test_paper_mode_simulates_orders_without_network() -> None:
    async with make_client(no_network, signer=None, trading_enabled=False) as client:
        assert client.is_paper
        assert await client.get_balance() == 100_000
        order = OrderRequest(ticker=TICKER, side="yes", count=10, max_cost_cents=455, kelly_fraction=0.1, reasoning="")
        result = await client.submit_order(order)
        assert result.paper
        assert result.status == "executed"
        assert result.filled_count == 10
        assert result.price_cents == 45
        assert await client.get_balance() == 100_000 - 450


async def test_paper_mode_rejects_insufficient_balance() -> None:
    client = KalshiClient(
        trading_enabled=False,
        paper_starting_balance_cents=100,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(no_network)),
    )
    order = OrderRequest(ticker=TICKER, side="no", count=3, max_cost_cents=150, kelly_fraction=0.1, reasoning="")
    async with client:
        with pytest.raises(PaperTradingError):
            await client.submit_order(order)
        assert await client.get_balance() == 100


async def test_paper_mode_cancel_rejects_unknown_and_filled_orders() -> None:
    async with make_client(no_network, signer=None, trading_enabled=False) as client:
        with pytest.raises(PaperTradingError) as excinfo:
            await client.cancel_order("nope")
        assert excinfo.value.status_code == 404

        order = OrderRequest(ticker=TICKER, side="yes", count=1, max_cost_cents=50, kelly_fraction=0.1, reasoning="")
        filled = await client.submit_order(order)
        with pytest.raises(PaperTradingError):
            await client.cancel_order(filled.order_id)


def test_from_env_defaults_to_paper_on_demo() -> None:
    client = KalshiClient.from_env({})
    assert client.is_paper
    assert client.base_url == DEMO_BASE_URL


def test_from_env_live_prod_with_key(private_key: rsa.RSAPrivateKey) -> None:
    pem = private_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    client = KalshiClient.from_env(
        {"TRADING_ENABLED": "true", "KALSHI_ENV": "prod", "KALSHI_API_KEY_ID": "kid", "KALSHI_PRIVATE_KEY": pem}
    )
    assert client.trading_enabled
    assert client.base_url == PROD_BASE_URL
    assert set(client.ws_auth_headers()) == {"KALSHI-ACCESS-KEY", "KALSHI-ACCESS-SIGNATURE", "KALSHI-ACCESS-TIMESTAMP"}


def test_from_env_live_without_credentials_fails() -> None:
    with pytest.raises(ValueError):
        KalshiClient.from_env({"TRADING_ENABLED": "true"})


# --------------------------------------------------------------------------- order book


def test_order_book_snapshot_delta_and_derived_asks() -> None:
    book = OrderBook(TICKER)
    book.apply_snapshot({"yes": [[38, 100], [40, 50]], "no": [[50, 10], [55, 30]]}, NOW)
    assert book.best_yes_bid == 40
    assert book.best_yes_ask == 45  # 100 - best NO bid (55)
    assert book.best_no_ask == 60
    assert book.imbalance == pytest.approx((50 - 30) / 80)

    book.apply_delta({"price": 55, "delta": -30, "side": "no"}, NOW)
    assert book.best_yes_ask == 50
    book.apply_delta({"price": 42, "delta": 5, "side": "yes"}, NOW)
    assert book.best_yes_bid == 42


def test_order_book_parses_dollar_fields() -> None:
    book = OrderBook(TICKER)
    book.apply_snapshot({"yes_dollars": [["0.3800", "100"]], "no_dollars": [["0.5500", "30"]]}, NOW)
    assert (book.best_yes_bid, book.best_yes_ask) == (38, 45)
    book.apply_delta({"price_dollars": "0.3900", "delta_fp": "7", "side": "yes"}, NOW)
    assert book.best_yes_bid == 39


def test_order_book_parses_production_fixed_point_snapshot() -> None:
    # Shapes captured from the production orderbook_delta channel on 2026-10-09.
    book = OrderBook(TICKER)
    book.apply_snapshot(
        {
            "yes_dollars_fp": [["0.0010", "643283.40"], ["0.0040", "27594.00"], ["0.2400", "44.30"], ["0.2500", "12.00"]],
            "no_dollars_fp": [["0.7300", "3051.00"], ["0.7400", "9.50"]],
        },
        NOW,
    )
    assert (book.best_yes_bid, book.best_yes_ask) == (25, 26)
    assert book.yes_bids[0] == pytest.approx(643283.40 + 27594.00)  # sub-penny levels share the 0c bucket

    # Fractional deltas must net to zero exactly, not leave a phantom level behind.
    book.apply_delta({"price_dollars": "0.2500", "delta_fp": "-4.85", "side": "yes"}, NOW)
    book.apply_delta({"price_dollars": "0.2500", "delta_fp": "-7.15", "side": "yes"}, NOW)
    assert book.best_yes_bid == 24
    book.apply_delta({"price_dollars": "0.2900", "delta_fp": "0.10", "side": "yes"}, NOW)
    assert book.best_yes_bid == 29  # "0.2900" must not float-round down to 28


def test_sub_penny_bids_round_down_so_asks_round_up() -> None:
    book = OrderBook(TICKER)
    book.apply_snapshot({"yes_dollars_fp": [["0.2450", "10"]], "no_dollars_fp": [["0.7450", "10"]]}, NOW)
    assert (book.best_yes_bid, book.best_yes_ask, book.best_no_ask) == (24, 26, 76)




def test_empty_order_book_has_no_quotes() -> None:
    book = OrderBook(TICKER)
    assert book.best_yes_ask is None and book.best_no_ask is None
    assert book.imbalance == 0.0


# --------------------------------------------------------------------------- scanner


class StubPricer:
    """Stands in for pricing.btc_digital_option; returns a fixed model probability."""

    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.calls: list[tuple[Btc15mMarketState, MarketQuote]] = []

    def __call__(self, state: Btc15mMarketState, quote: MarketQuote) -> FairValueOutput:
        self.calls.append((state, quote))
        return FairValueOutput(
            ticker=quote.ticker,
            strike_price=quote.strike_price,
            model_probability=self.probability,
            fair_cents=round(self.probability * 100),
            gross_edge=self.probability - quote.yes_ask / 100,
            volatility_used=state.historical_volatility,
            d2_score=0.0,
        )


class FakeKalshiClient:
    ws_url = "wss://demo-api.kalshi.co/trade-api/ws/v2"

    def __init__(self, markets: list[KalshiMarket]) -> None:
        self.markets = markets
        self.calls = 0

    async def get_markets(self, series_ticker: str = "KXBTC15M", status: str = "open") -> list[KalshiMarket]:
        assert series_ticker == "KXBTC15M"
        self.calls += 1
        return self.markets

    def ws_auth_headers(self) -> dict[str, str]:
        return {"KALSHI-ACCESS-KEY": "kid"}


class FakeWebSocket:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.incoming = [json.dumps(m) for m in messages]
        self.sent: list[dict[str, Any]] = []

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        if not self.incoming:
            raise ConnectionClosedOK(None, None)
        return self.incoming.pop(0)


class FakeConnector:
    def __init__(self, sockets: list[FakeWebSocket | Exception]) -> None:
        self.sockets = sockets
        self.connections: list[tuple[str, dict[str, str]]] = []

    @contextlib.asynccontextmanager
    async def __call__(self, url: str, headers: dict[str, str]) -> AsyncIterator[FakeWebSocket]:
        self.connections.append((url, headers))
        sock = self.sockets.pop(0)
        if isinstance(sock, Exception):
            raise sock
        yield sock


def market(ticker: str = TICKER, close: datetime = CLOSE) -> KalshiMarket:
    return KalshiMarket.from_api(raw_market(ticker, close))


def snapshot(yes: list[list[int]], no: list[list[int]], seq: int = 1, ticker: str = TICKER) -> dict[str, Any]:
    return {"type": "orderbook_snapshot", "sid": 1, "seq": seq, "msg": {"market_ticker": ticker, "yes": yes, "no": no}}


def delta(price: int, change: int, side: str, seq: int, ticker: str = TICKER) -> dict[str, Any]:
    return {
        "type": "orderbook_delta",
        "sid": 1,
        "seq": seq,
        "msg": {"market_ticker": ticker, "price": price, "delta": change, "side": side},
    }


SPOT = SpotContext(spot_price=82_700.0, historical_volatility=0.55, momentum_1m=0.08)


def make_scanner(
    probability: float,
    markets: list[KalshiMarket] | None = None,
    *,
    connector: FakeConnector | None = None,
    spot: SpotContext | None = SPOT,
    **kwargs: Any,
) -> tuple[KalshiOrderBookScanner, StubPricer]:
    pricer = StubPricer(probability)
    scanner = KalshiOrderBookScanner(
        FakeKalshiClient(markets if markets is not None else [market()]),  # type: ignore[arg-type]
        spot_provider=lambda: spot,
        fair_value_fn=pricer,
        params=StrategyParameters(min_gross_edge=0.035),
        connect=connector,
        clock=lambda: NOW,
        **kwargs,
    )
    return scanner, pricer


async def test_scanner_flags_underpriced_yes() -> None:
    scanner, pricer = make_scanner(probability=0.60)
    await scanner.refresh_markets()
    candidates = scanner.handle_message(snapshot(yes=[[40, 50]], no=[[55, 30]]))

    assert [(c.side, c.ask_cents) for c in candidates] == [("yes", 45)]
    assert candidates[0].edge == pytest.approx(0.15)
    state, quote = pricer.calls[0]
    assert (quote.yes_bid, quote.yes_ask, quote.strike_price) == (40, 45, 82500.0)
    assert state.seconds_remaining == 600
    assert state.current_spot_price == SPOT.spot_price
    assert state.historical_volatility == SPOT.historical_volatility
    assert state.order_book_imbalance == pytest.approx((50 - 30) / 80)


async def test_scanner_never_prices_a_crossed_book() -> None:
    # 50c YES bid against a 31c YES ask (69c NO bid): the desync seen live before the fp fix.
    scanner, pricer = make_scanner(probability=0.25)
    await scanner.refresh_markets()
    assert scanner.handle_message(snapshot(yes=[[50, 10]], no=[[69, 10]])) == []
    assert scanner.books[scanner.markets.popitem()[0]].is_crossed
    assert pricer.calls == []


async def test_scanner_flags_underpriced_no() -> None:
    scanner, _ = make_scanner(probability=0.20)
    await scanner.refresh_markets()
    candidates = scanner.handle_message(snapshot(yes=[[30, 10]], no=[[60, 10]]))
    # NO ask = 100 - 30 = 70c; P(no) = 0.80 → edge 0.10.
    assert [(c.side, c.ask_cents) for c in candidates] == [("no", 70)]
    assert candidates[0].model_probability == pytest.approx(0.80)
    assert candidates[0].edge == pytest.approx(0.10)


async def test_scanner_filters_edges_at_or_below_threshold() -> None:
    scanner, _ = make_scanner(probability=0.48)
    await scanner.refresh_markets()
    # YES edge 0.48 - 0.45 = 0.03, NO edge 0.52 - 0.60 < 0 → nothing passes 0.035.
    assert scanner.handle_message(snapshot(yes=[[40, 50]], no=[[55, 30]])) == []

    scanner.params = StrategyParameters(min_gross_edge=0.02)
    assert [c.side for c in scanner.evaluate(TICKER)] == ["yes"]


async def test_scanner_skips_without_spot_or_two_sided_book() -> None:
    scanner, pricer = make_scanner(probability=0.9, spot=None)
    await scanner.refresh_markets()
    assert scanner.handle_message(snapshot(yes=[[40, 50]], no=[[55, 30]])) == []

    scanner, pricer = make_scanner(probability=0.9)
    await scanner.refresh_markets()
    assert scanner.handle_message(snapshot(yes=[[40, 50]], no=[])) == []
    assert pricer.calls == []


async def test_scanner_drops_expired_markets_on_refresh() -> None:
    scanner, _ = make_scanner(0.6, markets=[market(), market("KXBTC15M-26OCT09-B80000", NOW - timedelta(seconds=1))])
    markets = await scanner.refresh_markets()
    assert [m.ticker for m in markets] == [TICKER]


async def test_scanner_suppresses_duplicate_candidates() -> None:
    scanner, _ = make_scanner(probability=0.60)
    await scanner.refresh_markets()
    assert len(scanner.handle_message(snapshot(yes=[[40, 50]], no=[[55, 30]], seq=1))) == 1
    # Size change at a non-top level leaves the ask unchanged → no repeat.
    assert scanner.handle_message(delta(50, 5, "no", seq=2)) == []
    # Ask moves 45 → 44 → new candidate.
    assert [c.ask_cents for c in scanner.handle_message(delta(56, 10, "no", seq=3))] == [44]


async def test_scanner_detects_sequence_gap() -> None:
    scanner, _ = make_scanner(probability=0.60)
    await scanner.refresh_markets()
    scanner.handle_message(snapshot(yes=[[40, 50]], no=[[55, 30]], seq=1))
    with pytest.raises(OrderBookOutOfSync):
        scanner.handle_message(delta(41, 1, "yes", seq=3))


async def test_scanner_raises_on_ws_error_message() -> None:
    scanner, _ = make_scanner(probability=0.60)
    await scanner.refresh_markets()
    with pytest.raises(KalshiWebSocketError):
        scanner.handle_message({"type": "error", "id": 1, "msg": {"code": 6, "msg": "Already subscribed"}})


async def test_run_session_subscribes_and_queues_candidates() -> None:
    ws = FakeWebSocket(
        [
            {"type": "subscribed", "id": 1, "msg": {"channel": "orderbook_delta", "sid": 1}},
            snapshot(yes=[[40, 50]], no=[[55, 30]], seq=1),
            delta(56, 10, "no", seq=2),
        ]
    )
    connector = FakeConnector([ws])
    seen: list[ScanCandidate] = []

    async def on_candidate(candidate: ScanCandidate) -> None:
        seen.append(candidate)

    scanner, _ = make_scanner(probability=0.60, connector=connector, on_candidate=on_candidate)
    with pytest.raises(ConnectionClosedOK):
        await scanner.run_session()

    assert connector.connections == [(FakeKalshiClient.ws_url, {"KALSHI-ACCESS-KEY": "kid"})]
    assert ws.sent == [
        {"id": 1, "cmd": "subscribe", "params": {"channels": ["orderbook_delta"], "market_tickers": [TICKER]}}
    ]
    assert [c.ask_cents for c in seen] == [45, 44]
    assert scanner.candidates.qsize() == 2
    assert scanner.candidates.get_nowait().ask_cents == 45


async def test_run_reconnects_after_failure_and_stops() -> None:
    ws = FakeWebSocket([snapshot(yes=[[40, 50]], no=[[55, 30]], seq=1)])
    connector = FakeConnector([OSError("connection refused"), ws])
    scanner, _ = make_scanner(probability=0.60, connector=connector, reconnect_delay=0.01)
    scanner._on_candidate = lambda _c: scanner.stop()

    await asyncio.wait_for(scanner.run(), timeout=2)

    assert len(connector.connections) == 2
    assert scanner.candidates.qsize() == 1


async def test_candidate_queue_drops_oldest_when_full() -> None:
    scanner, _ = make_scanner(probability=0.60, queue_maxsize=1)
    await scanner.refresh_markets()
    for candidate in scanner.handle_message(snapshot(yes=[[40, 50]], no=[[55, 30]], seq=1)):
        await scanner._emit(candidate)
    for candidate in scanner.handle_message(delta(56, 10, "no", seq=2)):
        await scanner._emit(candidate)
    assert scanner.candidates.qsize() == 1
    assert scanner.candidates.get_nowait().ask_cents == 44


def test_missing_pricing_module_gives_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pricing.btc_digital_option", None)
    with pytest.raises(ImportError, match="fair_value_fn"):
        KalshiOrderBookScanner(FakeKalshiClient([]), spot_provider=lambda: SPOT)  # type: ignore[arg-type]


def test_from_env_accepts_production_alias() -> None:
    client = KalshiClient.from_env({"KALSHI_ENV": "production"})
    assert client.base_url == "https://api.elections.kalshi.com/trade-api/v2"
    assert client.is_paper


def test_signer_accepts_single_line_pem_with_escaped_newlines() -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()
    ).decode()
    signer = KalshiRequestSigner.from_pem("key-id", pem.replace("\n", "\\n"))
    assert signer.headers("GET", "/trade-api/v2/markets")["KALSHI-ACCESS-KEY"] == "key-id"


async def test_scanner_reprices_only_when_top_of_book_moves_or_interval_elapses() -> None:
    now = [NOW]
    pricer = StubPricer(0.60)
    scanner = KalshiOrderBookScanner(
        FakeKalshiClient([market()]),  # type: ignore[arg-type]
        spot_provider=lambda: SPOT,
        fair_value_fn=pricer,
        params=StrategyParameters(min_gross_edge=0.035),
        clock=lambda: now[0],
    )
    await scanner.refresh_markets()
    scanner.handle_message(snapshot(yes=[[40, 50], [30, 10]], no=[[55, 30]]))
    assert len(pricer.calls) == 1

    scanner.handle_message(delta(30, 5, "yes", seq=2))  # below the touch: no re-price
    scanner.handle_message(delta(30, -3, "yes", seq=3))
    assert len(pricer.calls) == 1

    scanner.handle_message(delta(41, 5, "yes", seq=4))  # new best YES bid
    assert len(pricer.calls) == 2

    now[0] += timedelta(seconds=1.5)  # spot may have moved: periodic re-price
    scanner.handle_message(delta(30, 1, "yes", seq=5))
    assert len(pricer.calls) == 3
