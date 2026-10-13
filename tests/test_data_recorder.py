"""Research-data recording: index exchanges, high-resolution final minutes, book sizes, Kalshi tape."""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlmodel import select

from db.database import Database
from db.models import BtcPriceHistory, ContractOutcome, IndexSample, KalshiTapeBar, MarketSnapshot
from execution.kalshi_client import KalshiAPIError, PublicTrade
from services.exchange_recorder import (
    SAMPLE_FIELDS,
    Book,
    BookGap,
    CryptocomHandler,
    ExchangeRecorder,
    composite,
    handle_bitstamp,
    handle_bullish,
    handle_coinbase,
    handle_gemini,
    handle_kraken,
    rti_estimate,
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


def test_order_book_messages_build_each_venues_book() -> None:
    book = Book()
    assert handle_coinbase(
        {"type": "snapshot", "product_id": "BTC-USD", "bids": [["83073.68", "0.5"], ["83070.00", "1"]],
         "asks": [["83073.69", "0.2"]]}, book,
    )
    assert handle_coinbase(
        {"type": "l2update", "product_id": "BTC-USD", "changes": [["buy", "83073.68", "0.00000000"], ["sell", "83075.00", "2"]]},
        book,
    )
    assert book.best() == (83070.0, 83073.69) and book.asks == {83073.69: 0.2, 83075.0: 2.0}
    assert not handle_coinbase({"type": "subscriptions", "channels": []}, book)

    kraken = Book()
    handle_kraken(
        {"channel": "book", "type": "snapshot", "data": [{"symbol": "BTC/USD", "bids": [{"price": 83077.7, "qty": 1.0}],
                                                          "asks": [{"price": 83077.8, "qty": 3.0}]}]}, kraken,
    )
    handle_kraken({"channel": "book", "type": "update", "data": [{"bids": [{"price": 83077.7, "qty": 0.0}],
                                                                   "asks": [{"price": 83077.75, "qty": 0.1}]}]}, kraken)
    assert kraken.bids == {} and kraken.best() is None and min(kraken.asks) == 83077.75
    assert not handle_kraken({"channel": "heartbeat"}, kraken)

    bitstamp = Book()
    assert handle_bitstamp({"event": "data", "channel": "order_book_btcusd",
                            "data": {"bids": [["83072.76", "0.5"]], "asks": [["83072.77", "0.1"]]}}, bitstamp)
    assert bitstamp.best() == (83072.76, 83072.77)
    assert not handle_bitstamp({"event": "bts:subscription_succeeded", "channel": "order_book_btcusd", "data": {}}, bitstamp)

    gemini = Book()
    handle_gemini({"type": "l2_updates", "symbol": "BTCUSD", "changes": [["buy", "83077.26", "0.3"], ["sell", "83077.27", "1"]]}, gemini)
    handle_gemini({"type": "l2_updates", "symbol": "BTCUSD", "changes": [["sell", "83077.27", "0.0"], ["sell", "83078", "1"]]}, gemini)
    assert gemini.best() == (83077.26, 83078.0)
    assert not handle_gemini({"type": "trade", "symbol": "BTCUSD"}, gemini)

    bullish = Book()
    handle_bullish({"type": "snapshot", "dataType": "V1TALevel2",
                    "data": {"symbol": "BTCUSD", "bids": ["83299.5000", "0.03", "83299.4000", "0.004"],
                             "asks": ["83299.6000", "0.003"]}}, bullish)
    assert bullish.best() == (83299.5, 83299.6) and bullish.bids[83299.4] == 0.004

    crossed = Book()
    crossed.replace([(83080.0, 1.0)], [(83070.0, 1.0)])
    assert crossed.best() is None


def test_cryptocom_book_deltas_must_follow_the_sequence() -> None:
    handler, book = CryptocomHandler(), Book()
    snapshot = {"id": -1, "method": "subscribe", "result": {"channel": "book", "data": [
        {"bids": [["83078.48", "0.5", "3"]], "asks": [["83078.49", "0.2", "1"]], "u": 10}]}}
    assert handler(snapshot, book) and book.best() == (83078.48, 83078.49)
    update = {"id": -1, "result": {"channel": "book.update", "data": [
        {"update": {"bids": [["83078.48", "0", "0"], ["83078.40", "1", "1"]], "asks": []}, "u": 11, "pu": 10}]}}
    assert handler(update, book) and book.best() == (83078.40, 83078.49)
    gap = {"id": -1, "result": {"channel": "book.update", "data": [{"update": {"bids": [], "asks": []}, "u": 13, "pu": 12}]}}
    with pytest.raises(BookGap):
        handler(gap, book)


def test_depth_mid_looks_past_dust_at_the_top_of_the_book() -> None:
    book = Book()
    book.replace([(83_100.0, 0.0001), (83_000.0, 5.0)], [(83_101.0, 0.5), (83_103.0, 5.0)])
    assert book.mid() == 83_100.5
    bid = 83_100.0 * 0.0001 + 83_000.0 * 0.9999
    assert book.depth_mid(1.0) == pytest.approx((bid + 83_101.0 * 0.5 + 83_103.0 * 0.5) / 2)
    assert book.depth_mid(50.0) is None  # not enough size


def test_brti_estimate_weights_the_consolidated_book_near_the_top() -> None:
    a, b = Book(), Book()
    a.replace([(99.99, 1.0), (99.98, 5.0)], [(100.01, 1.0), (100.02, 5.0)])
    b.replace([(99.97, 2.0)], [(100.03, 2.0)])
    assert rti_estimate([a, b]) == pytest.approx(100.0)  # symmetric books: the mid
    skewed = Book()
    skewed.replace([(99.99, 10.0)], [(100.01, 0.1), (100.40, 10.0)])  # thin ask side
    assert 100.0 < rti_estimate([skewed]) < 100.2
    assert rti_estimate([Book()]) is None


def test_recorder_samples_every_index_proxy() -> None:
    clock = Clock(T0)
    rec = ExchangeRecorder(clock=clock)
    for name, mid in (("coinbase", 83_001.0), ("kraken", 83_005.0), ("bitstamp", 83_003.0)):
        rec.update(name, mid - 1, mid + 1)
    rec.update("gemini", 90_000.0, 90_002.0)  # a broken venue: >1% from the others, dropped
    sample = rec.sample()
    assert set(sample) <= {*SAMPLE_FIELDS, "sources"} and set(SAMPLE_FIELDS) <= set(IndexSample.model_fields)
    assert sample["sources"] == 3 and "gemini" not in sample
    assert sample["composite"] == 83_003.0 and sample["depth_composite"] == 83_003.0
    assert sample["kraken_depth"] == 83_005.0
    assert sample["brti_estimate"] is not None


def test_composite_is_the_median_of_fresh_mids() -> None:
    clock = Clock(T0)
    rec = ExchangeRecorder(clock=clock)
    assert rec.composite() is None
    rec.update("coinbase", 100.0, 102.0)  # mid 101
    rec.update("kraken", 104.0, 106.0)  # mid 105
    rec.update("gemini", 106.0, 108.0)
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
