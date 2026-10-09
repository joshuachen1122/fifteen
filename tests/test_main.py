"""Integration tests for the orchestration loop in main.py (paper client, in-memory DB, fake LLM)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest
from sqlmodel import select

from core.schemas import BtcSpotTick, FairValueOutput, LLMVerdict
from db.database import Database
from db.models import EvaluationTraceRecord, TradeRecord
from execution.kalshi_client import KalshiClient, KalshiMarket
from execution.order_scanner import OrderBook, ScanCandidate
from main import Settings, TradingAgent
from services.btc_spot_feed import BtcSpotFeed
from services.optimizer import Optimizer, ParameterGrid
from services.risk_engine import RiskEngine, kalshi_taker_fee_cents

T0 = datetime(2026, 10, 9, 14, 0, tzinfo=timezone.utc)
TICKER = "KXBTC15M-26OCT091415-B82500"
STRIKE = 82_500.0
YES_ASK = 45  # best NO bid 55
YES_BID = 40


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class FakeEvaluator:
    def __init__(self, should_bet: bool = True, probability_shift: float = 0.0) -> None:
        self.should_bet = should_bet
        self.probability_shift = probability_shift
        self.calls: list[tuple[FairValueOutput, dict[str, Any]]] = []

    async def evaluate(self, candidate: FairValueOutput, context: dict[str, Any]) -> LLMVerdict:
        self.calls.append((candidate, context))
        return LLMVerdict(
            should_bet=self.should_bet,
            adjusted_probability=min(max(candidate.model_probability + self.probability_shift, 0.01), 0.99),
            confidence_score=0.8,
            reasoning="approved" if self.should_bet else "macro news window",
            risk_signals=[] if self.should_bet else ["Macro news window"],
        )


class SettlingClient(KalshiClient):
    """Paper client whose settlement lookups are served from memory."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.results: dict[str, KalshiMarket] = {}

    async def get_market(self, ticker: str) -> KalshiMarket:
        return self.results[ticker]


@pytest.fixture
async def db():
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.init()
    yield database
    await database.dispose()


def warm_feed(clock: Clock, seconds: int = 301) -> BtcSpotFeed:
    feed = BtcSpotFeed()
    rng = np.random.default_rng(7)
    prices = 82_700 + rng.normal(0, 4, seconds)  # spot sits $200 above the strike
    start = clock.now - timedelta(seconds=seconds - 1)
    for i, price in enumerate(prices):
        feed.buffer.add_tick(BtcSpotTick(source="coinbase", price=float(price), timestamp=start + timedelta(seconds=i)))
    return feed


def advance(agent: TradingAgent, clock: Clock, delta: timedelta) -> None:
    """Move time forward while keeping the spot feed live."""
    end = clock.now + delta
    while clock.now < end:
        clock.now = min(clock.now + timedelta(seconds=1), end)
        agent.feed.buffer.add_tick(BtcSpotTick(source="coinbase", price=82_700.0, timestamp=clock.now))


def make_agent(db: Database, clock: Clock, evaluator: Any, **settings: Any) -> TradingAgent:
    client = SettlingClient()
    agent = TradingAgent(
        Settings(web_enabled=False, **settings), db=db, client=client, feed=warm_feed(clock), evaluator=evaluator, clock=clock
    )
    market = KalshiMarket(ticker=TICKER, strike_price=STRIKE, close_time=clock.now + timedelta(minutes=10))
    book = OrderBook(TICKER)
    book.apply_snapshot({"yes": [[YES_BID, 100]], "no": [[100 - YES_ASK, 120]]}, clock.now)
    agent.scanner.markets = {TICKER: market}
    agent.scanner.books = {TICKER: book}
    return agent


def yes_candidate(agent: TradingAgent) -> ScanCandidate:
    candidates = {c.side: c for c in agent.scanner.evaluate(TICKER)}
    assert "yes" in candidates, "fixture should present a YES edge"
    return candidates["yes"]


async def trade_rows(db: Database) -> list[TradeRecord]:
    async with db.session() as s:
        return list((await s.exec(select(TradeRecord))).all())


async def test_candidate_flows_through_llm_and_fee_aware_kelly_to_paper_fill(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator())
    await agent.restore()
    candidate = yes_candidate(agent)
    assert candidate.edge > agent.params.min_gross_edge

    await agent.handle_candidate(candidate)

    expected = RiskEngine(100_000, now=T0).size_order(
        candidate.fair_value, "yes", YES_ASK, agent.params, probability_override=candidate.fair_value.model_probability
    )
    assert expected is not None and expected.count > 0
    fee = kalshi_taker_fee_cents(expected.count, YES_ASK)

    [row] = await trade_rows(db)
    assert (row.side, row.count, row.filled_count, row.status) == ("yes", expected.count, expected.count, "filled")
    assert row.avg_fill_price_cents == YES_ASK and row.fees_cents == fee
    assert (row.strike_price, row.yes_bid, row.yes_ask, row.is_paper) == (STRIKE, YES_BID, YES_ASK, True)
    assert row.kelly_fraction <= agent.params.max_trade_bankroll_pct

    assert agent.risk is not None
    assert agent.risk.cash_cents == 100_000 - expected.count * YES_ASK - fee
    # The limit is the ask itself; the taker fee is not folded into the per-contract price.
    assert await agent.client.get_balance() == 100_000 - expected.count * YES_ASK

    async with db.session() as s:
        trace = (await s.exec(select(EvaluationTraceRecord))).one()
    assert trace.decision == "trade" and trace.id == row.evaluation_id

    # One position per contract: a fresh signal on the same ticker is ignored.
    advance(agent, clock, timedelta(minutes=2))
    await agent.handle_candidate(yes_candidate(agent))
    assert len(agent.evaluator.calls) == 1  # type: ignore[union-attr]


async def test_llm_veto_records_observation_without_trading(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator(should_bet=False))
    await agent.restore()

    await agent.handle_candidate(yes_candidate(agent))

    [row] = await trade_rows(db)
    assert row.side is None and row.status == "skipped" and row.yes_ask == YES_ASK
    assert agent.risk is not None and agent.risk.cash_cents == 100_000
    async with db.session() as s:
        trace = (await s.exec(select(EvaluationTraceRecord))).one()
    assert trace.decision == "skip" and trace.skip_reason and trace.skip_reason.startswith("llm_veto")
    assert trace.llm_risk_signals == ["Macro news window"]


async def test_llm_probability_cut_below_threshold_blocks_trade(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator(probability_shift=-0.5))
    await agent.restore()
    await agent.handle_candidate(yes_candidate(agent))
    [row] = await trade_rows(db)
    assert row.side is None
    assert agent.risk is not None and agent.risk.cash_cents == 100_000


async def test_missing_llm_fails_closed(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, evaluator=None)
    await agent.restore()
    await agent.handle_candidate(yes_candidate(agent))
    [row] = await trade_rows(db)
    assert row.side is None


async def test_cooldown_limits_llm_calls_per_ticker_side(db: Database) -> None:
    clock = Clock(T0)
    evaluator = FakeEvaluator(should_bet=False)
    agent = make_agent(db, clock, evaluator, llm_cooldown_sec=60)
    await agent.restore()

    await agent.handle_candidate(yes_candidate(agent))
    advance(agent, clock, timedelta(seconds=30))
    await agent.handle_candidate(yes_candidate(agent))
    assert len(evaluator.calls) == 1
    advance(agent, clock, timedelta(seconds=31))
    await agent.handle_candidate(yes_candidate(agent))
    assert len(evaluator.calls) == 2
    assert len(await trade_rows(db)) == 1  # re-evaluations update the ticker's single row


async def test_settlement_books_net_pnl_and_triggers_optimizer(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator(), optimize_every_n_settled=1)
    optimizer_runs: list[Optimizer] = []

    def small_optimizer(engine: Any) -> Optimizer:
        opt = Optimizer(
            engine, grid=ParameterGrid(min_gross_edge=[0.03], volatility_lookback_sec=[300], momentum_weight=[0.15])
        )
        optimizer_runs.append(opt)
        return opt

    agent.make_optimizer = small_optimizer  # type: ignore[method-assign]
    await agent.restore()
    await agent.handle_candidate(yes_candidate(agent))
    [row] = await trade_rows(db)
    count, fee = row.filled_count, row.fees_cents

    # Not yet expired: nothing to settle.
    await agent.poll_once()
    assert (await trade_rows(db))[0].settled_yes is None and not optimizer_runs

    clock.now = T0 + timedelta(minutes=11)
    assert isinstance(agent.client, SettlingClient)
    agent.client.results[TICKER] = KalshiMarket(
        ticker=TICKER, strike_price=STRIKE, close_time=T0 + timedelta(minutes=10), result="yes", settlement_price=82_710.5
    )
    await agent.poll_once()

    [settled] = await trade_rows(db)
    expected_pnl = count * 100 - count * YES_ASK - fee
    assert (settled.settled_yes, settled.status, settled.realized_pnl_cents) == (True, "settled", expected_pnl)
    assert settled.settlement_price == 82_710.5 and settled.settled_at == clock.now
    assert agent.risk is not None
    assert agent.risk.realized_pnl_cents == expected_pnl
    assert agent.risk.equity_cents == 100_000 + expected_pnl
    assert await agent.client.get_balance() == 100_000 + count * (100 - YES_ASK)
    assert not agent._positions

    assert len(optimizer_runs) == 1 and agent.settled_since_optimization == 0
    assert agent.dashboard.optimizer_status == "done"


async def test_unsettled_market_is_retried_later(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator())
    await agent.restore()
    await agent.handle_candidate(yes_candidate(agent))
    clock.now = T0 + timedelta(minutes=11)
    assert isinstance(agent.client, SettlingClient)
    agent.client.results[TICKER] = KalshiMarket(ticker=TICKER, strike_price=STRIKE, close_time=T0, result="")
    assert await agent.settle_completed() == 0
    assert len(agent._positions) == 1


async def test_nightly_optimizer_runs_once_per_day(db: Database) -> None:
    clock = Clock(T0.replace(hour=4))
    agent = make_agent(db, clock, FakeEvaluator(), nightly_optimize_hour_utc=4)
    runs: list[str] = []

    async def fake_run(reason: str) -> None:
        runs.append(reason)

    agent.run_optimizer = fake_run  # type: ignore[method-assign]
    await agent.restore()
    await agent.poll_once()
    clock.now += timedelta(minutes=3)
    await agent.poll_once()
    assert runs == ["nightly"]


async def test_restore_rebuilds_open_positions_and_bankroll(db: Database) -> None:
    clock = Clock(T0)
    first = make_agent(db, clock, FakeEvaluator())
    await first.restore()
    await first.handle_candidate(yes_candidate(first))
    assert first.risk is not None

    clock.now += timedelta(minutes=1)
    second = make_agent(db, clock, FakeEvaluator())
    await second.restore()
    assert second.risk is not None
    assert len(second._positions) == 1
    assert second.risk.equity_cents == first.risk.equity_cents
    assert second.risk.cash_cents == first.risk.cash_cents
    assert await second.client.get_balance() == second.risk.cash_cents
    assert second._has_position(TICKER)
