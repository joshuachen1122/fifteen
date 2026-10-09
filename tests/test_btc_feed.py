from __future__ import annotations

import asyncio
import json
import math
import statistics
import threading
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
import websockets
from sqlmodel import select

from core.constants import SECONDS_PER_YEAR, SPOT_BUFFER_WINDOW_SEC
from core.schemas import BtcSpotTick, MarketQuote, StrategyParameters
from core.state_manager import BtcStateManager
from db.database import Database
from db.models import BtcPriceHistory, EvaluationTraceRecord, StrategyParametersRecord, TradeRecord
from services import btc_spot_feed
from services.btc_spot_feed import (
    BtcSpotFeed,
    SpotPriceBuffer,
    parse_binance_message,
    parse_coinbase_message,
)

T0 = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)


def tick(sec: float, price: float, source: str = "coinbase") -> BtcSpotTick:
    return BtcSpotTick(source=source, price=price, timestamp=T0 + timedelta(seconds=sec))


def fill(buffer: SpotPriceBuffer, prices: list[float], start: int = 0) -> None:
    for i, p in enumerate(prices):
        buffer.add_tick(tick(start + i, p))


def zigzag(n: int, base: float = 60_000.0, step: float = 0.0005) -> list[float]:
    """Prices whose 1s log returns alternate +step / -step."""
    return [base * math.exp(step if i % 2 else 0.0) for i in range(n)]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


class TestParsers:
    def test_coinbase_match(self) -> None:
        t = parse_coinbase_message(
            {"type": "match", "product_id": "BTC-USD", "price": "62000.50", "time": "2026-10-09T12:00:01.123456Z"}
        )
        assert t is not None
        assert t.source == "coinbase"
        assert t.price == 62000.50
        assert t.timestamp == datetime(2026, 10, 9, 12, 0, 1, 123456, tzinfo=timezone.utc)

    def test_coinbase_last_match_accepted(self) -> None:
        msg = {"type": "last_match", "product_id": "BTC-USD", "price": "1", "time": "2026-10-09T12:00:00Z"}
        assert parse_coinbase_message(msg) is not None

    @pytest.mark.parametrize(
        "msg",
        [
            {"type": "subscriptions", "channels": []},
            {"type": "match", "product_id": "ETH-USD", "price": "3000", "time": "2026-10-09T12:00:00Z"},
            {"type": "match", "product_id": "BTC-USD", "price": "nan", "time": "2026-10-09T12:00:00Z"},
            {"type": "match", "product_id": "BTC-USD", "price": "-5", "time": "2026-10-09T12:00:00Z"},
            {"type": "match", "product_id": "BTC-USD", "price": "62000"},
            {"type": "match", "product_id": "BTC-USD", "price": "62000", "time": "garbage"},
        ],
    )
    def test_coinbase_rejects(self, msg: dict[str, object]) -> None:
        assert parse_coinbase_message(msg) is None

    def test_binance_raw_and_combined(self) -> None:
        raw = {"e": "trade", "s": "BTCUSD", "p": "61999.99", "q": "0.1", "T": 1_791_547_200_500}
        t = parse_binance_message(raw)
        assert t is not None
        assert t.source == "binance"
        assert t.price == 61999.99
        assert t.timestamp == datetime.fromtimestamp(1_791_547_200.5, tz=timezone.utc)
        assert parse_binance_message({"stream": "btcusd@trade", "data": raw}) == t

    @pytest.mark.parametrize("msg", [{"e": "aggTrade", "p": "1", "T": 1}, {"result": None, "id": 1}, {"e": "trade"}])
    def test_binance_rejects(self, msg: dict[str, object]) -> None:
        assert parse_binance_message(msg) is None


# ---------------------------------------------------------------------------
# SpotPriceBuffer
# ---------------------------------------------------------------------------


class TestSpotPriceBuffer:
    def test_empty(self) -> None:
        b = SpotPriceBuffer()
        assert len(b) == 0
        assert b.latest_price() is None
        assert b.volatility() is None
        assert b.momentum() is None
        assert b.is_stale(T0)

    def test_same_second_keeps_last_price(self) -> None:
        b = SpotPriceBuffer()
        b.add_tick(tick(0.1, 100.0))
        b.add_tick(tick(0.9, 101.0))
        assert len(b) == 1
        assert b.latest_price() == 101.0

    def test_gap_is_forward_filled(self) -> None:
        b = SpotPriceBuffer()
        b.add_tick(tick(0, 100.0))
        b.add_tick(tick(5, 105.0))
        assert b.snapshot() == [100.0] * 5 + [105.0]

    def test_out_of_order_tick_rejected(self) -> None:
        b = SpotPriceBuffer()
        b.add_tick(tick(10, 100.0))
        assert b.add_tick(tick(9, 999.0)) is False
        assert b.snapshot() == [100.0]

    def test_rolling_window_is_15_minutes(self) -> None:
        b = SpotPriceBuffer()
        fill(b, [float(i + 1) for i in range(SPOT_BUFFER_WINDOW_SEC + 100)])
        assert len(b) == SPOT_BUFFER_WINDOW_SEC == 900
        snap = b.snapshot()
        assert snap[0] == 101.0
        assert snap[-1] == float(SPOT_BUFFER_WINDOW_SEC + 100)

    def test_gap_longer_than_window_resets(self) -> None:
        b = SpotPriceBuffer(window_sec=10)
        fill(b, [1.0, 2.0, 3.0])
        b.add_tick(tick(1000, 50.0))
        assert b.snapshot() == [50.0]

    def test_naive_timestamp_treated_as_utc(self) -> None:
        b = SpotPriceBuffer()
        b.add_tick(BtcSpotTick(source="x", price=1.0, timestamp=T0.replace(tzinfo=None)))
        last = b.last_tick
        assert last is not None and last.timestamp == T0

    def test_volatility_matches_reference(self) -> None:
        b = SpotPriceBuffer()
        prices = zigzag(400)
        fill(b, prices)
        window = prices[-301:]
        rets = [math.log(window[i + 1] / window[i]) for i in range(300)]
        expected = statistics.stdev(rets) * math.sqrt(SECONDS_PER_YEAR)
        vol = b.volatility(300)
        assert vol is not None
        assert vol == pytest.approx(expected, rel=1e-9)
        # sanity: 5 bps per second alternating -> roughly 280% annualized
        assert 2.0 < vol < 3.5

    def test_volatility_constant_price_is_zero(self) -> None:
        b = SpotPriceBuffer()
        fill(b, [60_000.0] * 120)
        assert b.volatility(300) == 0.0

    def test_volatility_requires_min_samples(self) -> None:
        b = SpotPriceBuffer()
        fill(b, zigzag(10))
        assert b.volatility(300, min_samples=30) is None
        assert b.volatility(300, min_samples=5) is not None

    def test_volatility_uses_only_lookback(self) -> None:
        b = SpotPriceBuffer()
        # 600s of turbulence then 301s of flat prices
        fill(b, zigzag(600, step=0.01) + [60_000.0] * 301)
        assert b.volatility(300) == 0.0
        vol_15m = b.volatility(899)
        assert vol_15m is not None and vol_15m > 0

    def test_momentum_1m_percent(self) -> None:
        b = SpotPriceBuffer()
        fill(b, [100.0 + 0.1 * i for i in range(90)])
        # now = 108.9, 60s ago = 102.9
        assert b.momentum(60) == pytest.approx((108.9 / 102.9 - 1) * 100)

    def test_momentum_requires_full_window(self) -> None:
        b = SpotPriceBuffer()
        fill(b, [100.0] * 60)
        assert b.momentum(60) is None
        b.add_tick(tick(60, 101.0))
        assert b.momentum(60) == pytest.approx(1.0)

    def test_staleness(self) -> None:
        b = SpotPriceBuffer()
        b.add_tick(tick(0, 100.0))
        assert not b.is_stale(T0 + timedelta(seconds=5))
        assert b.is_stale(T0 + timedelta(seconds=11))

    def test_concurrent_writers_and_readers(self) -> None:
        b = SpotPriceBuffer(window_sec=300)
        errors: list[BaseException] = []
        stop = threading.Event()

        def writer(offset: int) -> None:
            try:
                for i in range(2000):
                    b.add_tick(tick(i, 60_000.0 + ((i + offset) % 7)))
            except BaseException as e:  # pragma: no cover - surfaced via assertion
                errors.append(e)

        def reader() -> None:
            try:
                while not stop.is_set():
                    b.volatility(299, min_samples=2)
                    b.momentum(60)
                    assert len(b) <= 300
            except BaseException as e:  # pragma: no cover
                errors.append(e)

        readers = [threading.Thread(target=reader) for _ in range(4)]
        writers = [threading.Thread(target=writer, args=(k,)) for k in range(4)]
        for t in readers + writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in readers:
            t.join()
        assert errors == []
        assert len(b) == 300


# ---------------------------------------------------------------------------
# BtcSpotFeed (against a local WebSocket server)
# ---------------------------------------------------------------------------


def coinbase_match(sec: int, price: float) -> str:
    ts = (T0 + timedelta(seconds=sec)).isoformat().replace("+00:00", "Z")
    return json.dumps({"type": "match", "product_id": "BTC-USD", "price": str(price), "time": ts})


async def wait_for(predicate: object, timeout: float = 3.0) -> None:
    async def _poll() -> None:
        while not predicate():  # type: ignore[operator]
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout)


class TestBtcSpotFeed:
    def test_unknown_source(self) -> None:
        with pytest.raises(ValueError):
            BtcSpotFeed("kraken")

    def test_handle_raw_message(self) -> None:
        feed = BtcSpotFeed("coinbase")
        seen: list[BtcSpotTick] = []
        feed.add_listener(lambda t: (_ for _ in ()).throw(RuntimeError("boom")))  # bad listener is isolated
        feed.add_listener(seen.append)
        assert feed.handle_raw_message("not json") is None
        assert feed.handle_raw_message("[1, 2]") is None
        assert feed.handle_raw_message(json.dumps({"type": "heartbeat"})) is None
        t = feed.handle_raw_message(coinbase_match(0, 61000.0))
        assert t is not None and seen == [t]
        assert feed.latest_price() == 61000.0

    @pytest.mark.asyncio
    async def test_subscribes_and_buffers_ticks(self) -> None:
        received: list[dict[str, object]] = []

        async def handler(ws: websockets.ServerConnection) -> None:
            received.append(json.loads(await ws.recv()))
            await ws.send(json.dumps({"type": "subscriptions", "channels": [{"name": "matches"}]}))
            for i in range(5):
                await ws.send(coinbase_match(i, 60_000.0 + i))
            await ws.wait_closed()

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            feed = BtcSpotFeed("coinbase", url=f"ws://127.0.0.1:{port}")
            feed.start()
            try:
                await wait_for(lambda: len(feed.buffer) == 5)
            finally:
                await feed.stop()

        assert received == [{"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["matches"]}]
        assert feed.buffer.snapshot() == [60_000.0, 60_001.0, 60_002.0, 60_003.0, 60_004.0]
        assert not feed.connected.is_set()

    @pytest.mark.asyncio
    async def test_reconnects_after_disconnect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(btc_spot_feed, "WS_RECONNECT_INITIAL_SEC", 0.01)
        connections = 0

        async def handler(ws: websockets.ServerConnection) -> None:
            nonlocal connections
            connections += 1
            await ws.send(json.dumps({"e": "trade", "p": str(60_000 + connections), "T": 1_791_547_200_000 + connections * 1000}))
            await ws.close()  # drop every connection after one trade

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            feed = BtcSpotFeed("binance", url=f"ws://127.0.0.1:{port}")
            feed.start()
            try:
                await wait_for(lambda: connections >= 3 and len(feed.buffer) >= 3)
            finally:
                await feed.stop()

        assert feed.buffer.snapshot()[:3] == [60_001.0, 60_002.0, 60_003.0]


# ---------------------------------------------------------------------------
# BtcStateManager
# ---------------------------------------------------------------------------

TICKER = "KXBTC15M-26OCT09-B82500"
NOW = T0 + timedelta(seconds=119)


@pytest.fixture
def warm_buffer() -> SpotPriceBuffer:
    b = SpotPriceBuffer()
    fill(b, zigzag(120, base=82_000.0))  # last tick at NOW
    return b


@pytest.fixture
def manager(warm_buffer: SpotPriceBuffer) -> BtcStateManager:
    return BtcStateManager(warm_buffer, clock=lambda: NOW)


class TestBtcStateManager:
    @pytest.mark.asyncio
    async def test_merges_spot_with_market(self, manager: BtcStateManager, warm_buffer: SpotPriceBuffer) -> None:
        expiry = NOW + timedelta(seconds=421, milliseconds=600)
        await manager.upsert_market(TICKER, 82_500.0, expiry)
        await manager.update_order_book_imbalance(TICKER, 0.25)

        state = await manager.get_market_state(TICKER)
        assert state is not None
        assert state.ticker == TICKER
        assert state.strike_price == 82_500.0
        assert state.expiration_time == expiry
        assert state.seconds_remaining == 421
        assert state.current_spot_price == warm_buffer.latest_price()
        assert state.historical_volatility == warm_buffer.volatility(300)
        assert state.momentum_1m == warm_buffer.momentum(60)
        assert state.order_book_imbalance == 0.25

    @pytest.mark.asyncio
    async def test_upsert_updates_existing_and_keeps_imbalance(self, manager: BtcStateManager) -> None:
        await manager.upsert_market(TICKER, 82_500.0, NOW + timedelta(minutes=5))
        await manager.update_order_book_imbalance(TICKER, -0.4)
        await manager.upsert_market(TICKER, 82_750.0, NOW + timedelta(minutes=6))
        state = await manager.get_market_state(TICKER)
        assert state is not None
        assert state.strike_price == 82_750.0
        assert state.order_book_imbalance == -0.4
        assert manager.active_tickers() == [TICKER]

    @pytest.mark.asyncio
    async def test_quote_is_attached(self, manager: BtcStateManager) -> None:
        await manager.upsert_market(TICKER, 82_500.0, NOW + timedelta(minutes=5))
        q = MarketQuote(ticker=TICKER, strike_price=82_500.0, yes_bid=40, yes_ask=43, volume=10, updated_at=NOW)
        await manager.update_quote(q)
        market = manager.get_market(TICKER)
        assert market is not None and market.quote == q

    @pytest.mark.asyncio
    async def test_validation(self, manager: BtcStateManager) -> None:
        with pytest.raises(ValueError):
            await manager.upsert_market("KXETH15M-26OCT09-B3000", 3000.0, NOW)
        with pytest.raises(ValueError):
            await manager.upsert_market(TICKER, 0.0, NOW)
        with pytest.raises(KeyError):
            await manager.update_order_book_imbalance(TICKER, 0.0)
        await manager.upsert_market(TICKER, 82_500.0, NOW + timedelta(minutes=5))
        with pytest.raises(ValueError):
            await manager.update_order_book_imbalance(TICKER, 1.5)

    @pytest.mark.asyncio
    async def test_expired_market_excluded_and_pruned(self, manager: BtcStateManager) -> None:
        live = "KXBTC15M-26OCT09-B83000"
        await manager.upsert_market(TICKER, 82_500.0, NOW)  # expires exactly now
        await manager.upsert_market(live, 83_000.0, NOW + timedelta(minutes=10))
        assert await manager.get_market_state(TICKER) is None
        assert [s.ticker for s in await manager.get_all_market_states()] == [live]
        assert await manager.prune_expired() == [TICKER]
        assert manager.active_tickers() == [live]

    @pytest.mark.asyncio
    async def test_all_states_sorted_by_expiry(self, manager: BtcStateManager) -> None:
        later, sooner = "KXBTC15M-26OCT09-B1", "KXBTC15M-26OCT09-B2"
        await manager.upsert_market(later, 82_000.0, NOW + timedelta(minutes=14))
        await manager.upsert_market(sooner, 82_000.0, NOW + timedelta(minutes=2))
        assert [s.ticker for s in await manager.get_all_market_states()] == [sooner, later]

    @pytest.mark.asyncio
    async def test_stale_spot_yields_no_state(self, warm_buffer: SpotPriceBuffer) -> None:
        m = BtcStateManager(warm_buffer, clock=lambda: NOW + timedelta(seconds=30))
        await m.upsert_market(TICKER, 82_500.0, NOW + timedelta(minutes=10))
        assert await m.get_market_state(TICKER) is None
        assert await m.get_all_market_states() == []

    @pytest.mark.asyncio
    async def test_cold_buffer_yields_no_state(self) -> None:
        b = SpotPriceBuffer()
        fill(b, [82_000.0] * 20)  # < 60s: no momentum, < 30 returns: no vol
        m = BtcStateManager(b, clock=lambda: T0 + timedelta(seconds=19))
        await m.upsert_market(TICKER, 82_500.0, T0 + timedelta(minutes=10))
        assert await m.get_market_state(TICKER) is None
        assert await m.get_market_state("KXBTC15M-unknown") is None

    @pytest.mark.asyncio
    async def test_volatility_lookback_follows_parameters(self, warm_buffer: SpotPriceBuffer) -> None:
        m = BtcStateManager(warm_buffer, clock=lambda: NOW)
        await m.upsert_market(TICKER, 82_500.0, NOW + timedelta(minutes=10))
        await m.update_parameters(StrategyParameters(volatility_lookback_sec=60))
        state = await m.get_market_state(TICKER)
        assert state is not None
        assert state.historical_volatility == warm_buffer.volatility(60)

    @pytest.mark.asyncio
    async def test_accepts_feed_as_spot_source(self, warm_buffer: SpotPriceBuffer) -> None:
        feed = BtcSpotFeed("coinbase", buffer=warm_buffer)
        m = BtcStateManager(feed, clock=lambda: NOW)
        await m.upsert_market(TICKER, 82_500.0, NOW + timedelta(minutes=10))
        assert await m.get_market_state(TICKER) is not None

    @pytest.mark.asyncio
    async def test_thread_safe_snapshots_during_updates(self, manager: BtcStateManager) -> None:
        tickers = [f"KXBTC15M-26OCT09-B{82_000 + i}" for i in range(50)]
        for t in tickers:
            await manager.upsert_market(t, 82_000.0, NOW + timedelta(minutes=10))
        errors: list[BaseException] = []
        stop = threading.Event()

        def reader() -> None:
            try:
                while not stop.is_set():
                    for s in manager.snapshot_all_market_states():
                        assert -1.0 <= s.order_book_imbalance <= 1.0
            except BaseException as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for th in threads:
            th.start()
        for i in range(500):
            t = tickers[i % len(tickers)]
            await manager.update_order_book_imbalance(t, (i % 21) / 10 - 1.0)
            if i % 50 == 0:
                await manager.remove_market(t)
                await manager.upsert_market(t, 82_000.0, NOW + timedelta(minutes=10))
            await asyncio.sleep(0)
        stop.set()
        for th in threads:
            th.join()
        assert errors == []
        assert len(manager.active_tickers()) == 50


# ---------------------------------------------------------------------------
# Database models
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db() -> Database:
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.init()
    yield database  # type: ignore[misc]
    await database.dispose()


class TestDatabase:
    @pytest.mark.asyncio
    async def test_price_history_roundtrip_is_utc(self, db: Database) -> None:
        pacific = timezone(timedelta(hours=-7))
        async with db.session() as s:
            s.add(BtcPriceHistory(source="coinbase", price=62_000.0, timestamp=T0.astimezone(pacific)))
            await s.commit()
        async with db.session() as s:
            row = (await s.exec(select(BtcPriceHistory))).one()
        assert row.timestamp == T0
        assert row.timestamp.tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_evaluation_trade_and_parameters(self, db: Database) -> None:
        async with db.session() as s:
            params = StrategyParametersRecord.from_schema(StrategyParameters(), is_active=True, source="default")
            s.add(params)
            await s.commit()
            await s.refresh(params)

            ev = EvaluationTraceRecord(
                ticker=TICKER,
                strike_price=82_500.0,
                expiration_time=T0 + timedelta(minutes=15),
                seconds_remaining=600,
                current_spot_price=82_400.0,
                historical_volatility=0.45,
                momentum_1m=0.02,
                order_book_imbalance=0.1,
                yes_bid=40,
                yes_ask=43,
                model_probability=0.47,
                fair_cents=47,
                gross_edge=0.04,
                volatility_used=0.45,
                d2_score=-0.08,
                llm_should_bet=True,
                llm_adjusted_probability=0.46,
                llm_confidence_score=0.7,
                llm_reasoning="edge holds",
                llm_risk_signals=["High volatility spike detected", "Macro news window"],
                decision="trade",
                strategy_parameters_id=params.id,
            )
            s.add(ev)
            await s.commit()
            await s.refresh(ev)

            trade = TradeRecord(
                evaluation_id=ev.id,
                ticker=TICKER,
                side="yes",
                count=5,
                max_cost_cents=215,
                kelly_fraction=0.02,
                reasoning="edge holds",
            )
            s.add(trade)
            await s.commit()

        async with db.session() as s:
            ev_row = (await s.exec(select(EvaluationTraceRecord))).one()
            trade_row = (await s.exec(select(TradeRecord).where(TradeRecord.evaluation_id == ev_row.id))).one()
            params_row = (await s.exec(select(StrategyParametersRecord).where(StrategyParametersRecord.is_active))).one()

        assert ev_row.llm_risk_signals == ["High volatility spike detected", "Macro news window"]
        assert ev_row.expiration_time == T0 + timedelta(minutes=15)
        assert ev_row.created_at.tzinfo == timezone.utc
        assert trade_row.status == "pending"
        assert trade_row.is_paper is True
        assert trade_row.settled_at is None
        assert params_row.to_schema() == StrategyParameters()

    @pytest.mark.asyncio
    async def test_foreign_keys_enforced(self, db: Database) -> None:
        from sqlalchemy.exc import IntegrityError

        async with db.session() as s:
            s.add(TradeRecord(evaluation_id=999, ticker=TICKER, side="no", count=1, max_cost_cents=50, kelly_fraction=0.01, reasoning="x"))
            with pytest.raises(IntegrityError):
                await s.commit()
