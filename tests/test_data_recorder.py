"""Research-data recording: index exchanges, high-resolution final minutes, book sizes, Kalshi tape."""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlmodel import select

from db.database import Database
from db.models import BtcPriceHistory, ContractOutcome, IndexSample, KalshiTapeBar, MarketSnapshot
from execution.kalshi_client import KalshiAPIError, PublicTrade
from services.exchange_recorder import (
    ExchangeRecorder,
    GeminiParser,
    composite,
    parse_bitstamp,
    parse_bullish,
    parse_coinbase,
    parse_cryptocom,
    parse_kraken,
)
from services.market_tape import tape_bars
from tests.test_main import STRIKE, T0, TICKER, YES_ASK, YES_BID, Clock, FakeEvaluator, SettlingClient, advance, make_agent


@pytest.fixture
async def db():
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.init()
    yield database
    await database.dispose()


# ---------------------------------------------------------------- exchange parsers (real message shapes)


def test_each_exchange_message_parses_to_best_bid_and_ask() -> None:
    assert parse_coinbase(
        {"type": "ticker", "product_id": "BTC-USD", "price": "83073.68", "best_bid": "83073.68", "best_ask": "83073.69"}
    ) == (83073.68, 83073.69)
    assert parse_coinbase({"type": "subscriptions", "channels": []}) is None
    assert parse_kraken(
        {"channel": "ticker", "type": "snapshot", "data": [{"symbol": "BTC/USD", "bid": 83077.7, "ask": 83077.8}]}
    ) == (83077.7, 83077.8)
    assert parse_kraken({"channel": "heartbeat"}) is None
    assert parse_bitstamp(
        {
            "data": {"bids": [["83072.76", "0.5"], ["83070.10", "0.006"]], "asks": [["83072.77", "0.1"]]},
            "channel": "order_book_btcusd",
            "event": "data",
        }
    ) == (83072.76, 83072.77)
    assert parse_bitstamp({"event": "bts:subscription_succeeded", "channel": "order_book_btcusd", "data": {}}) is None
    assert parse_cryptocom(
        {"id": -1, "method": "subscribe", "result": {"channel": "ticker", "data": [{"b": "83078.48", "k": "83078.49"}]}}
    ) == (83078.48, 83078.49)
    assert parse_bullish(
        {"type": "update", "dataType": "V1TATickerResponse", "data": {"bestBid": "83062.8", "bestAsk": "83062.9"}}
    ) == (83062.8, 83062.9)
    assert parse_kraken({"channel": "ticker", "data": [{"bid": 83080.0, "ask": 83070.0}]}) is None  # crossed

    gemini = GeminiParser()
    assert gemini({"type": "update", "events": [{"type": "change", "side": "ask", "price": "83077.27"}]}) is None
    assert gemini({"type": "update", "events": [{"type": "change", "side": "bid", "price": "83077.26"}]}) == (
        83077.26,
        83077.27,
    )


def test_composite_is_the_median_of_fresh_mids() -> None:
    clock = Clock(T0)
    rec = ExchangeRecorder(clock=clock)
    assert rec.composite() is None
    rec.update("coinbase", 100.0, 102.0)  # mid 101
    rec.update("kraken", 104.0, 106.0)  # mid 105
    rec.update("gemini", 9_000.0, 9_002.0)  # a broken venue barely moves a median
    assert rec.composite() == 105.0
    clock.now = T0 + timedelta(seconds=11)
    rec.update("kraken", 106.0, 108.0)
    assert rec.mids() == {"kraken": 107.0}  # the others went stale
    assert composite({}) is None
    assert {s["source"]: s["fresh"] for s in rec.status()}["coinbase"] is False


# ---------------------------------------------------------------- tape bars


def _trade(sec: float, yes: float, count: float, taker: str = "yes") -> PublicTrade:
    return PublicTrade(
        ticker=TICKER, created_time=T0 + timedelta(seconds=sec), yes_price_cents=yes, no_price_cents=100 - yes,
        count=count, taker_side=taker,  # type: ignore[arg-type]
    )


def test_tape_bars_aggregate_trades_per_second() -> None:
    trades = [_trade(0.1, 40, 10), _trade(0.9, 44, 30, "no"), _trade(2.5, 50, 5), _trade(9.0, 60, 1)]
    bars = tape_bars(TICKER, trades, T0, T0 + timedelta(seconds=5))
    assert [(b.start, b.trades, b.volume, b.taker_yes_volume) for b in bars] == [
        (T0, 2, 40, 10),
        (T0 + timedelta(seconds=2), 1, 5, 5),
    ]
    assert bars[0].vwap_yes_cents == pytest.approx((40 * 10 + 44 * 30) / 40)
    assert (bars[0].low_yes_cents, bars[0].high_yes_cents) == (40, 44)


# ---------------------------------------------------------------- agent recording


async def test_snapshots_go_to_one_per_second_in_the_final_two_minutes_with_book_and_index_data(db: Database) -> None:
    clock = Clock(T0)
    rec = ExchangeRecorder(clock=clock)
    agent = make_agent(db, clock, FakeEvaluator(), close_in=timedelta(minutes=3), exchanges=rec)
    await agent.restore()

    async def tick() -> int:
        advance(agent, clock, timedelta(seconds=1))
        rec.update("kraken", 82_705.0, 82_707.0)
        rec.update("bitstamp", 82_703.0, 82_705.0)
        await agent.sync_markets()
        return await agent.write_snapshots()

    early = [await tick() for _ in range(10)]  # 179s..170s left: every 5 seconds
    assert sum(early) == 2
    clock.now = T0 + timedelta(minutes=3) - timedelta(seconds=121)
    late = [await tick() for _ in range(70)]  # 120s..51s left: every second
    assert late == [1] * 70

    async with db.session() as s:
        snaps = (await s.exec(select(MarketSnapshot).order_by(MarketSnapshot.ts))).all()
    last = snaps[-1]
    assert last.seconds_remaining == 51
    assert (last.yes_bid_size, last.yes_ask_size) == (100, 120)  # fixture book: YES bid 100, NO bid 120
    assert (last.bid_depth_5c, last.ask_depth_5c) == (100, 120)
    assert last.composite_spot == pytest.approx(82_705.0)  # median of the two mids
    # Nine seconds into the settlement average: its observed mean is recorded.
    assert last.observed_seconds == 9 and last.observed_average == pytest.approx(82_700.0 + agent.spot.index_gap)
    assert snaps[0].observed_average is None


async def test_spot_and_index_samples_are_persisted_every_second_only_near_a_close(db: Database) -> None:
    clock = Clock(T0)
    rec = ExchangeRecorder(clock=clock)
    agent = make_agent(db, clock, FakeEvaluator(), close_in=timedelta(minutes=10), exchanges=rec)
    await agent.restore()
    saved, index = None, None
    for _ in range(10):  # 10 minutes left: every 5 seconds
        advance(agent, clock, timedelta(seconds=1))
        agent._latest_tick = agent.feed.buffer.last_tick
        rec.update("coinbase", 82_699.0, 82_701.0)
        saved, index = await agent.persist_prices(saved, index)
    clock.now = T0 + timedelta(minutes=10) - timedelta(seconds=60)
    for _ in range(10):  # final minute: every second
        advance(agent, clock, timedelta(seconds=1))
        agent._latest_tick = agent.feed.buffer.last_tick
        rec.update("coinbase", 82_699.0, 82_701.0)
        saved, index = await agent.persist_prices(saved, index)

    async with db.session() as s:
        spot = (await s.exec(select(BtcPriceHistory))).all()
        samples = (await s.exec(select(IndexSample))).all()
    assert len(spot) == 2 + 10 and len(samples) == 2 + 10
    assert samples[-1].coinbase == 82_700.0 and samples[-1].composite == 82_700.0 and samples[-1].sources == 1


async def test_tape_bars_are_stored_once_per_settled_contract(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator())
    await agent.restore()
    close = T0 + timedelta(minutes=10)
    assert isinstance(agent.client, SettlingClient)
    agent.client.trades = [
        PublicTrade(ticker=TICKER, created_time=close - timedelta(seconds=s), yes_price_cents=float(YES_ASK),
                    no_price_cents=float(100 - YES_ASK), count=3, taker_side="yes")
        for s in (200, 90, 30, 30)  # the first is outside the 180s window
    ]
    async with db.session() as s:
        s.add(ContractOutcome(ticker=TICKER, strike_price=STRIKE, close_time=close, settled_yes=False))
        await s.commit()
    clock.now = close + timedelta(minutes=2)

    assert await agent.record_tapes() == 1
    assert await agent.record_tapes() == 0  # fetched once
    async with db.session() as s:
        bars = (await s.exec(select(KalshiTapeBar).order_by(KalshiTapeBar.start))).all()
        outcome = await s.get(ContractOutcome, TICKER)
    assert [(b.trades, b.volume) for b in bars] == [(1, 3), (2, 6)]
    assert outcome is not None and outcome.tape_bars == 2
    assert YES_BID < bars[0].vwap_yes_cents == YES_ASK


async def test_outcome_without_tape_access_is_retried(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator())
    await agent.restore()

    async def failing(*_: object, **__: object) -> list[PublicTrade]:
        raise KalshiAPIError("rate limited")

    agent.client.get_trades = failing  # type: ignore[method-assign]
    async with db.session() as s:
        s.add(ContractOutcome(ticker=TICKER, strike_price=STRIKE, close_time=T0, settled_yes=True))
        await s.commit()
    assert await agent.record_tapes() == 0
    async with db.session() as s:
        outcome = await s.get(ContractOutcome, TICKER)
    assert outcome is not None and outcome.tape_bars is None


# ---------------------------------------------------------------- training export


async def test_training_frame_labels_snapshots_and_joins_index_and_tape(tmp_path) -> None:
    from services.dataset import build_training_frame, export

    path = tmp_path / "agent.db"
    database = Database(f"sqlite+aiosqlite:///{path}")
    await database.init()
    close = T0 + timedelta(minutes=1)
    snap = dict(ticker=TICKER, strike_price=STRIKE, close_time=close, yes_bid=YES_BID, yes_ask=YES_ASK,
                spot_price=82_705.0, historical_volatility=0.2, momentum_1m=0.0, order_book_imbalance=0.0,
                model_probability=0.9, index_gap=5.0)
    async with database.session() as s:
        s.add(MarketSnapshot(ts=T0 + timedelta(seconds=10, milliseconds=300), seconds_remaining=50, **snap))
        s.add(MarketSnapshot(ts=T0 + timedelta(seconds=30), seconds_remaining=30, **snap))  # no index sample near
        s.add(MarketSnapshot(ts=T0, seconds_remaining=60, **(snap | {"ticker": "UNSETTLED"})))
        s.add(IndexSample(ts=T0 + timedelta(seconds=9), kraken=82_710.0, gemini=82_712.0, composite=82_711.0, sources=2))
        s.add(KalshiTapeBar(ticker=TICKER, start=T0 + timedelta(seconds=10), trades=4, volume=20.0,
                            taker_yes_volume=15.0, vwap_yes_cents=79.5, low_yes_cents=79.0, high_yes_cents=80.0))
        s.add(ContractOutcome(ticker=TICKER, strike_price=STRIKE, close_time=close, settled_yes=True,
                              settlement_price=82_720.0))
        await s.commit()
    await database.dispose()

    df = build_training_frame(path)
    assert list(df["seconds_remaining"]) == [50, 30]  # unsettled contracts are left out
    first, second = df.iloc[0], df.iloc[1]
    assert first["label"] == 1 and first["mid_prob"] == pytest.approx((YES_BID + YES_ASK) / 200)
    assert first["coinbase_raw"] == 82_700.0 and first["spot_minus_strike"] == pytest.approx(205.0)
    assert first["index_composite"] == 82_711.0 and first["kraken"] == 82_710.0
    assert first["composite_minus_strike"] == pytest.approx(211.0)
    assert first["tape_volume"] == 20.0 and first["tape_taker_yes_volume"] == 15.0
    assert pd_isna(second["index_composite"]) and pd_isna(second["tape_volume"])  # nothing within 2s

    counts = export(path, tmp_path / "out")
    assert counts["training"] == 2 and counts["index_samples"] == 1 and (tmp_path / "out" / "training.csv.gz").exists()


def pd_isna(value: object) -> bool:
    import pandas as pd

    return bool(pd.isna(value))


async def test_data_panel_reports_recording_health(db: Database) -> None:
    import httpx

    from web.app import create_app

    clock = Clock(T0)
    rec = ExchangeRecorder(clock=clock)
    rec.update("kraken", 82_709.0, 82_711.0)
    agent = make_agent(db, clock, FakeEvaluator(), exchanges=rec)
    await agent.restore()
    await agent.sync_markets()
    await agent.write_snapshots()
    await agent.refresh_data_health()
    health = agent.dashboard.data_health
    assert health["last_hour"]["snapshots"] == 1 and health["composite"] == 82_710.0

    app = create_app(agent.dashboard, db.engine)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        page = (await http.get("/partials/data")).text
        index = (await http.get("/")).text
    assert "Research data recorder" in page and "kraken" in page and "down" in page  # never connected: down
    assert "1 market snapshots" in page and "Research data recorder" in index
