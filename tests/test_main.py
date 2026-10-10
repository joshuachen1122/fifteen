"""Integration tests for the orchestration loop in main.py (paper client, in-memory DB, fake LLM)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest
from sqlmodel import select

from core.schemas import BtcSpotTick, FairValueOutput, LLMVerdict
from db.database import Database
from db.models import (
    ContractOutcome,
    EvaluationReportRecord,
    EvaluationTraceRecord,
    MakerShadowOrder,
    MarketSnapshot,
    TradeRecord,
)
from execution.kalshi_client import KalshiClient, KalshiMarket, PublicTrade
from execution.order_scanner import OrderBook, ScanCandidate
from main import Settings, TradingAgent
from services.btc_spot_feed import BtcSpotFeed
from services.llm_evaluator import LLMUsage
from services.optimizer import Optimizer, ParameterGrid
from services.risk_engine import RiskEngine, kalshi_taker_fee_cents

T0 = datetime(2026, 10, 9, 14, 0, tzinfo=timezone.utc)
TICKER = "KXBTC15M-26OCT091415-B82500"
STRIKE = 82_500.0
YES_ASK = 80  # best NO bid 20; the model says ~95% YES, so a ~15-point edge
YES_BID = 76


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


# Token counts measured on the deployed evaluator (Sonnet 5.5, effort medium).
CALL_USAGE = LLMUsage(model="claude-sonnet-5-5", input_tokens=1590, output_tokens=290)


class FakeEvaluator:
    def __init__(self, should_bet: bool = True, probability_shift: float = 0.0) -> None:
        self.should_bet = should_bet
        self.probability_shift = probability_shift
        self.calls: list[tuple[FairValueOutput, dict[str, Any]]] = []

    async def evaluate_with_usage(
        self, candidate: FairValueOutput, context: dict[str, Any]
    ) -> tuple[LLMVerdict, LLMUsage]:
        self.calls.append((candidate, context))
        verdict = LLMVerdict(
            should_bet=self.should_bet,
            adjusted_probability=min(max(candidate.model_probability + self.probability_shift, 0.01), 0.99),
            confidence_score=0.8,
            reasoning="approved" if self.should_bet else "macro news window",
            risk_signals=[] if self.should_bet else ["Macro news window"],
        )
        return verdict, CALL_USAGE


class SettlingClient(KalshiClient):
    """Paper client whose settlement, series and trade-tape lookups are served from memory."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.results: dict[str, KalshiMarket] = {}
        self.trades: list[PublicTrade] = []
        self.fee_type = "quadratic"

    async def get_market(self, ticker: str) -> KalshiMarket:
        return self.results[ticker]

    async def get_series(self, series_ticker: str) -> dict[str, Any]:
        return {"ticker": series_ticker, "fee_type": self.fee_type, "exchange_index": 2}

    async def get_trades(self, ticker: str, *, min_ts: datetime, max_ts: datetime, **_: Any) -> list[PublicTrade]:
        return [t for t in self.trades if t.ticker == ticker and min_ts <= t.created_time <= max_ts]


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

    expected = RiskEngine(100_000, now=T0).size_order(  # paper sizing is uncapped
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
    agent = make_agent(db, clock, evaluator, llm_cooldown_sec=60, llm_reask_edge_step=-1.0)  # isolate the cooldown
    await agent.restore()

    await agent.handle_candidate(yes_candidate(agent))
    advance(agent, clock, timedelta(seconds=30))
    await agent.handle_candidate(yes_candidate(agent))
    assert len(evaluator.calls) == 1
    advance(agent, clock, timedelta(seconds=31))
    await agent.handle_candidate(yes_candidate(agent))
    assert len(evaluator.calls) == 2
    assert len(await trade_rows(db)) == 1  # re-evaluations update the ticker's single row


async def test_settlement_books_net_pnl_and_records_the_outcome(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator())
    await agent.restore()
    await agent.handle_candidate(yes_candidate(agent))
    [row] = await trade_rows(db)
    count, fee = row.filled_count, row.fees_cents

    # Not yet expired: nothing to settle.
    await agent.poll_once()
    assert (await trade_rows(db))[0].settled_yes is None

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

    async with db.session() as session:
        outcome = await session.get(ContractOutcome, TICKER)
    assert outcome is not None and outcome.settled_yes is True and outcome.settlement_price == 82_710.5


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


async def test_llm_usage_is_recorded_per_evaluation(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator(should_bet=False))
    await agent.restore()
    await agent.handle_candidate(yes_candidate(agent))
    async with db.session() as s:
        trace = (await s.exec(select(EvaluationTraceRecord))).one()
    assert (trace.llm_model, trace.llm_input_tokens, trace.llm_output_tokens) == ("claude-sonnet-5-5", 1590, 290)
    assert trace.llm_cost_usd == pytest.approx(CALL_USAGE.cost_usd)
    assert agent.llm_spend_today(clock.now) == pytest.approx(CALL_USAGE.cost_usd)


async def test_vetoed_candidate_is_not_re_asked_without_a_material_edge_change(db: Database) -> None:
    clock = Clock(T0)
    evaluator = FakeEvaluator(should_bet=False)
    agent = make_agent(db, clock, evaluator, llm_cooldown_sec=0)
    await agent.restore()
    first = yes_candidate(agent)
    await agent.handle_candidate(first)
    advance(agent, clock, timedelta(seconds=5))
    await agent.handle_candidate(yes_candidate(agent))
    assert len(evaluator.calls) == 1 and agent.llm_screened["vetoed_no_material_change"] == 1

    better = first.model_copy(update={"edge": first.edge + 0.03})
    await agent.handle_candidate(better)
    assert len(evaluator.calls) == 2


async def test_per_market_llm_call_cap(db: Database) -> None:
    clock = Clock(T0)
    evaluator = FakeEvaluator(should_bet=False)
    agent = make_agent(db, clock, evaluator, llm_cooldown_sec=0, llm_max_calls_per_market=2, llm_reask_edge_step=0)
    await agent.restore()
    for _ in range(4):
        await agent.handle_candidate(yes_candidate(agent))
    assert len(evaluator.calls) == 2 and agent.llm_screened["market_call_cap"] == 2


async def test_implausible_edges_and_near_expiry_are_screened_for_free(db: Database) -> None:
    clock = Clock(T0)
    evaluator = FakeEvaluator()
    agent = make_agent(db, clock, evaluator)
    await agent.restore()
    c = yes_candidate(agent)
    await agent.handle_candidate(c.model_copy(update={"edge": 0.45}))
    late = c.market_state.model_copy(update={"seconds_remaining": 45})
    await agent.handle_candidate(c.model_copy(update={"market_state": late}))
    assert evaluator.calls == []
    assert agent.llm_screened == {"edge_implausible": 1, "too_close_to_expiry": 1}
    assert await trade_rows(db) == []


async def test_daily_llm_budget_fails_closed_and_resets_next_day(db: Database) -> None:
    clock = Clock(T0)
    evaluator = FakeEvaluator()
    agent = make_agent(db, clock, evaluator, llm_daily_budget_usd=CALL_USAGE.cost_usd * 0.5)
    await agent.restore()
    await agent.handle_candidate(yes_candidate(agent))  # first call is allowed, then the budget is spent
    assert len(evaluator.calls) == 1

    other = make_agent(db, clock, evaluator, llm_daily_budget_usd=CALL_USAGE.cost_usd * 0.5)
    await other.restore()  # spend survives a restart
    assert other.llm_spend_today(clock.now) == pytest.approx(CALL_USAGE.cost_usd)
    candidate = yes_candidate(other).model_copy(update={"ticker": "KXBTC15M-26OCT091430-B82500"})
    await other.handle_candidate(candidate)
    assert len(evaluator.calls) == 1 and other.llm_screened["daily_budget_exhausted"] == 1

    clock.now += timedelta(days=1)
    assert other.llm_spend_today(clock.now) == 0.0


async def test_dashboard_stats_summarize_pnl_bets_and_llm_spend(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator())
    await agent.restore()
    await agent.handle_candidate(yes_candidate(agent))
    [row] = await trade_rows(db)

    clock.now = T0 + timedelta(minutes=11)
    assert isinstance(agent.client, SettlingClient)
    agent.client.results[TICKER] = KalshiMarket(ticker=TICKER, strike_price=STRIKE, close_time=T0, result="yes")
    await agent.settle_completed()
    agent.llm_screened["edge_implausible"] += 3
    await agent.refresh_stats()

    stats = agent.dashboard.stats
    assert stats is not None
    pnl = row.filled_count * (100 - YES_ASK) - row.fees_cents
    assert stats.total_pnl == pytest.approx(pnl / 100)
    assert (stats.bets_made, stats.bets_settled, stats.wins, stats.win_rate_pct) == (1, 1, 1, 100.0)
    assert stats.return_pct == pytest.approx(pnl / 100_000 * 100)
    assert (stats.llm_calls, stats.llm_approvals, stats.approval_rate_pct) == (1, 1, 100.0)
    assert stats.llm_cost_total == pytest.approx(CALL_USAGE.cost_usd)
    assert stats.pnl_after_llm == pytest.approx(pnl / 100 - CALL_USAGE.cost_usd)
    assert stats.candidates_screened == 3
    # One call in an 11-minute window, extrapolated to 30 days.
    assert stats.projected_monthly_cost == pytest.approx(CALL_USAGE.cost_usd / (11 / 60) * 720, rel=1e-3)


async def test_init_adds_new_columns_to_an_existing_database(tmp_path) -> None:
    import sqlite3

    new_columns = ["llm_model", "llm_input_tokens", "llm_output_tokens", "llm_cache_read_tokens",
                   "llm_cache_write_tokens", "llm_cost_usd"]
    path = tmp_path / "old.db"
    url = f"sqlite+aiosqlite:///{path}"
    database = Database(url)
    await database.init()
    await database.dispose()
    with sqlite3.connect(path) as conn:  # rewind evaluation_traces to the previous release's schema
        for column in new_columns:
            conn.execute(f"ALTER TABLE evaluation_traces DROP COLUMN {column}")
        conn.execute(
            "INSERT INTO evaluation_traces (created_at, ticker, strike_price, expiration_time, seconds_remaining,"
            " current_spot_price, historical_volatility, momentum_1m, order_book_imbalance, model_probability,"
            " fair_cents, gross_edge, volatility_used, d2_score, llm_risk_signals, decision)"
            " VALUES ('2026-10-09 19:40:54', 'KXBTC15M-OLD', 82500, '2026-10-09 19:45:00', 250, 82450, 0.27,"
            " 0.0, 0.0, 0.28, 28, 0.22, 0.27, -0.6, '[]', 'skip')"
        )

    database = Database(url)
    await database.init()
    await database.dispose()
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(evaluation_traces)")}
        assert set(new_columns) <= columns
        assert conn.execute("SELECT ticker, llm_cost_usd FROM evaluation_traces").fetchall() == [("KXBTC15M-OLD", None)]


async def test_stats_panel_renders(db: Database) -> None:
    import httpx

    from web.app import create_app

    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator())
    await agent.restore()
    app = create_app(agent.dashboard, db.engine)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        empty = await http.get("/partials/stats")
        assert empty.status_code == 200 and "Collecting stats" in empty.text

        await agent.handle_candidate(yes_candidate(agent))
        agent.llm_screened["vetoed_no_material_change"] += 2
        await agent.refresh_stats()
        page = await http.get("/partials/stats")
    assert page.status_code == 200
    for text in ("Total PnL", "Bets made", "LLM calls", "Projected / month", "PAPER", "vetoed no material change"):
        assert text in page.text


async def test_activity_panel_shows_positions_fills_and_llm_verdicts(db: Database) -> None:
    import httpx

    from web.app import create_app

    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator())
    await agent.restore()
    await agent.handle_candidate(yes_candidate(agent))  # approved and filled
    vetoing = make_agent(db, clock, FakeEvaluator(should_bet=False))
    await vetoing.restore()
    other = yes_candidate(vetoing).model_copy(update={"ticker": "KXBTC15M-26OCT091430-B82500"})
    await vetoing.handle_candidate(other)
    await agent.refresh_stats()

    activity = agent.dashboard.activity
    assert activity is not None
    [position] = activity.positions
    assert (position.ticker, position.side, position.mark_cents) == (TICKER, "yes", YES_BID)
    [row] = [r for r in await trade_rows(db) if r.side]
    assert position.unrealized_pnl == pytest.approx(((YES_BID - YES_ASK) * row.filled_count - row.fees_cents) / 100)
    assert [o.status for o in activity.orders] == ["filled"]
    assert [(d.side, d.ask_cents, d.decision, d.approved) for d in activity.decisions] == [
        ("yes", YES_ASK, "skip", False),
        ("yes", YES_ASK, "trade", True),
    ]

    app = create_app(agent.dashboard, db.engine)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        page = await http.get("/partials/activity")
    assert page.status_code == 200
    for text in ("Open positions (1)", "✓ traded", "✕ vetoed", "macro news window", "Macro news window", TICKER):
        assert text in page.text


async def test_stats_count_multiple_llm_approvals(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator())
    await agent.restore()
    await agent.handle_candidate(yes_candidate(agent))  # approved and traded
    for ticker in ("KXBTC15M-26OCT091430-B82500", "KXBTC15M-26OCT091445-B82500"):
        await agent.handle_candidate(yes_candidate(agent).model_copy(update={"ticker": ticker}))  # approved
    vetoing = make_agent(db, clock, FakeEvaluator(should_bet=False))
    await vetoing.restore()
    await vetoing.handle_candidate(yes_candidate(vetoing).model_copy(update={"ticker": "KXBTC15M-26OCT091500-B82500"}))
    await agent.refresh_stats()

    stats = agent.dashboard.stats
    assert stats is not None
    assert (stats.llm_calls, stats.llm_approvals) == (4, 3)
    assert stats.approval_rate_pct == pytest.approx(75.0)
    traded = sum(1 for d in (agent.dashboard.activity.decisions if agent.dashboard.activity else []) if d.decision == "trade")
    assert stats.bet_per_call_pct == pytest.approx(traded / 4 * 100)


# ---------------------------------------------------------------- real-money controls

from cryptography.hazmat.primitives.asymmetric import rsa as _rsa  # noqa: E402

from execution.kalshi_client import KalshiRequestSigner, OrderResult  # noqa: E402
from services.risk_engine import kalshi_taker_fee_cents as _fee  # noqa: E402


class FakeLiveClient(SettlingClient):
    """Credentialed client whose live endpoints are served from memory."""

    def __init__(self, shard_balance_cents: int = 2829, fill_price: float | None = None, fee: int | None = None) -> None:
        key = _rsa.generate_private_key(public_exponent=65537, key_size=2048)
        super().__init__(signer=KalshiRequestSigner("kid", key))
        self.shard_balance_cents = shard_balance_cents
        self.fill_price, self.fee = fill_price, fee
        self.live_orders: list[tuple[Any, int | None]] = []

    async def get_series_exchange_index(self, series_ticker: str) -> int | None:
        return 2

    async def get_balance(self, exchange_index: int | None = None) -> int:
        if self.is_paper:
            return await super().get_balance()
        assert exchange_index == 2, "live sizing must use the series' exchange shard"
        return self.shard_balance_cents

    async def submit_order(self, order: Any, *, client_order_id: str | None = None, exchange_index: int | None = None) -> OrderResult:
        if self.is_paper:
            return await super().submit_order(order, client_order_id=client_order_id)
        self.live_orders.append((order, exchange_index))
        price = order.max_cost_cents // order.count
        return OrderResult(
            order_id="live-1", ticker=order.ticker, side=order.side, status="executed", count=order.count,
            price_cents=price, filled_count=order.count,
            avg_fill_price_cents=self.fill_price if self.fill_price is not None else float(price),
            fees_cents=self.fee if self.fee is not None else _fee(order.count, price),
        )


def make_live_agent(db: Database, clock: Clock, client: FakeLiveClient, **settings: Any) -> TradingAgent:
    defaults = {"allow_live_trading": True, "dashboard_password": "pw"}
    agent = make_agent(db, clock, FakeEvaluator(), **{**defaults, **settings})
    client.results = agent.client.results  # type: ignore[attr-defined]
    agent.client = client
    agent.scanner._client = client
    return agent


async def test_paper_trades_are_uncapped_but_live_trades_respect_the_per_trade_cap(db: Database) -> None:
    clock = Clock(T0)
    paper = make_agent(db, clock, FakeEvaluator())
    await paper.restore()
    assert paper.risk is not None and paper.risk.max_order_cost_cents is None
    await paper.handle_candidate(yes_candidate(paper))
    [paper_row] = [r for r in await trade_rows(db) if r.side]
    assert paper_row.filled_count * YES_ASK + paper_row.fees_cents > 100

    live = make_live_agent(db, clock, FakeLiveClient(shard_balance_cents=100_000))
    await live.load_controls()
    await live.restore()
    await live.switch_mode(True)
    await live.handle_candidate(yes_candidate(live))  # same contract paper already traded
    [live_row] = [r for r in await trade_rows(db) if r.side and not r.is_paper]
    assert set(live._trade_rows.values()) == {live_row.id} and len(live._positions) == 1
    assert 1 <= live_row.filled_count and live_row.filled_count * YES_ASK + live_row.fees_cents <= 100


async def test_live_daily_cap_blocks_trades_once_spent(db: Database) -> None:
    clock = Clock(T0)
    agent = make_live_agent(db, clock, FakeLiveClient(shard_balance_cents=100_000), max_daily_trade_usd=1.00)
    await agent.load_controls()
    await agent.restore()
    await agent.switch_mode(True)
    await agent.handle_candidate(yes_candidate(agent))
    other = yes_candidate(agent).model_copy(update={"ticker": "KXBTC15M-26OCT091430-B82500"})
    await agent.handle_candidate(other)
    traded = [r for r in await trade_rows(db) if r.side]
    assert len(traded) == 1
    assert agent.risk is not None and agent.risk.remaining_daily_spend_cents(clock.now) < YES_ASK

    restarted = make_live_agent(db, clock, FakeLiveClient(shard_balance_cents=100_000), max_daily_trade_usd=1.00)
    await restarted.load_controls()
    await restarted.restore()  # today's live spend survives a restart
    assert restarted.risk is not None and restarted.risk.spent_today_cents == agent.risk.spent_today_cents


@pytest.mark.parametrize(
    ("settings", "with_signer", "reason"),
    [
        ({"allow_live_trading": False, "dashboard_password": "pw"}, True, "ALLOW_LIVE_TRADING"),
        ({"allow_live_trading": True, "dashboard_password": None}, True, "DASHBOARD_PASSWORD"),
        ({"allow_live_trading": True, "dashboard_password": "pw"}, False, "credentials"),
    ],
)
async def test_live_mode_is_blocked_without_every_safeguard(db: Database, settings, with_signer, reason) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator(), **settings)
    if with_signer:
        live = FakeLiveClient()
        agent.client = live
    await agent.restore()
    with pytest.raises(ValueError, match=reason):
        await agent.switch_mode(True)
    assert agent.client.is_paper


async def test_switch_to_live_sizes_from_the_exchange_shard_and_books_actual_fills(db: Database) -> None:
    clock = Clock(T0)
    live = FakeLiveClient(shard_balance_cents=2829, fill_price=79.4, fee=1)
    agent = make_live_agent(db, clock, live)
    await agent.load_controls()
    await agent.restore()
    assert agent.client.is_paper

    await agent.switch_mode(True)
    assert agent.is_live and agent.exchange_index == 2
    assert agent.risk is not None and agent.risk.equity_cents == 2829

    await agent.handle_candidate(yes_candidate(agent))
    [(order, exchange_index)] = live.live_orders
    assert exchange_index == 2 and order.max_cost_cents == order.count * YES_ASK  # limit at the ask
    [row] = [r for r in await trade_rows(db) if r.side]
    assert (row.is_paper, row.avg_fill_price_cents, row.fees_cents) == (False, 79.4, 1)
    assert agent.risk.cash_cents == 2829 - row.filled_count * 80 - 1  # 79.4c booked conservatively at 80c

    # The mode persists across restarts.
    again = make_live_agent(db, clock, FakeLiveClient())
    await again.load_controls()
    assert again.is_live


async def test_live_restart_is_refused_if_the_server_guard_is_removed(db: Database) -> None:
    clock = Clock(T0)
    agent = make_live_agent(db, clock, FakeLiveClient())
    await agent.load_controls()
    await agent.restore()
    await agent.switch_mode(True)

    guarded = make_live_agent(db, clock, FakeLiveClient(), allow_live_trading=False)
    await guarded.load_controls()
    assert guarded.client.is_paper


async def test_positions_opened_before_a_mode_switch_still_settle_with_pnl(db: Database) -> None:
    clock = Clock(T0)
    agent = make_live_agent(db, clock, FakeLiveClient())
    await agent.load_controls()
    await agent.restore()
    await agent.handle_candidate(yes_candidate(agent))  # paper fill
    [paper_row] = [r for r in await trade_rows(db) if r.side]
    await agent.switch_mode(True)
    assert not agent._positions  # the live book starts empty

    clock.now = T0 + timedelta(minutes=11)
    agent.client.results[TICKER] = KalshiMarket(ticker=TICKER, strike_price=STRIKE, close_time=T0, result="yes")  # type: ignore[attr-defined]
    assert await agent.settle_completed() == 1
    [settled] = [r for r in await trade_rows(db) if r.side]
    expected = paper_row.filled_count * (100 - YES_ASK) - paper_row.fees_cents
    assert (settled.status, settled.realized_pnl_cents, settled.is_paper) == ("settled", expected, True)


async def test_limits_can_be_changed_at_runtime_and_persist(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator())
    await agent.restore()
    await agent.set_limits(2.50, 40.00)
    assert agent.risk is not None and agent.risk.max_order_cost_cents is None  # paper stays uncapped
    with pytest.raises(ValueError):
        await agent.set_limits(5.00, 1.00)

    restarted = make_live_agent(db, clock, FakeLiveClient())
    await restarted.load_controls()
    assert (restarted.max_trade_cents, restarted.max_daily_cents) == (250, 4000)
    await restarted.restore()
    await restarted.switch_mode(True)
    assert restarted.risk is not None
    assert (restarted.risk.max_order_cost_cents, restarted.risk.daily_spend_limit_cents) == (250, 4000)


async def test_dashboard_is_password_protected_and_drives_the_controls(db: Database) -> None:
    import httpx

    from web.app import create_app

    clock = Clock(T0)
    agent = make_live_agent(db, clock, FakeLiveClient())
    await agent.load_controls()
    await agent.restore()
    app = create_app(agent.dashboard, db.engine, controls=agent, password="s3cret")
    app.get("/healthz")(lambda: {"ok": True})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as anon:
        assert (await anon.get("/")).status_code == 401
        assert (await anon.post("/controls/mode", data={"mode": "live"})).status_code == 401
        assert (await anon.get("/healthz")).status_code == 200
    assert agent.client.is_paper

    async with httpx.AsyncClient(transport=transport, base_url="http://t", auth=("me", "wrong")) as bad:
        assert (await bad.get("/")).status_code == 401

    async with httpx.AsyncClient(transport=transport, base_url="http://t", auth=("me", "s3cret")) as http:
        page = await http.get("/")
        assert page.status_code == 200 and "Trading controls" in page.text and "Switch to LIVE" in page.text
        switched = await http.post("/controls/mode", data={"mode": "live"})
        assert "Switched to LIVE trading" in switched.text and agent.is_live
        caps = await http.post("/controls/limits", data={"max_trade_usd": "0.75", "max_daily_trade_usd": "20"})
        assert "Spend caps updated" in caps.text and agent.max_trade_cents == 75
        bad_caps = await http.post("/controls/limits", data={"max_trade_usd": "9", "max_daily_trade_usd": "1"})
        assert "Invalid limits" in bad_caps.text and agent.max_trade_cents == 75
        back = await http.post("/controls/mode", data={"mode": "paper"})
        assert "Switched to PAPER" in back.text and agent.client.is_paper


async def test_controls_are_read_only_without_a_password(db: Database) -> None:
    import httpx

    from web.app import create_app

    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator())
    await agent.restore()
    app = create_app(agent.dashboard, db.engine, controls=agent, password=None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
        response = await http.post("/controls/mode", data={"mode": "live"})
    assert "Set DASHBOARD_PASSWORD" in response.text and agent.client.is_paper


async def test_sub_penny_live_fill_is_booked_at_one_cent_and_recorded_exactly(db: Database) -> None:
    clock = Clock(T0)
    live = FakeLiveClient(fill_price=0.2, fee=1)
    agent = make_live_agent(db, clock, live)
    await agent.load_controls()
    await agent.restore()
    await agent.switch_mode(True)
    await agent.handle_candidate(yes_candidate(agent))
    [position] = agent._positions.values()
    assert position.price_cents == 1  # never 0c
    [row] = [r for r in await trade_rows(db) if r.side]
    assert row.avg_fill_price_cents == pytest.approx(0.2)


async def test_dashboard_shows_only_the_current_modes_activity(db: Database) -> None:
    clock = Clock(T0)
    agent = make_live_agent(db, clock, FakeLiveClient(shard_balance_cents=100_000))
    await agent.load_controls()
    await agent.restore()
    await agent.handle_candidate(yes_candidate(agent))  # paper: LLM call + trade
    await agent.refresh_stats()
    assert agent.dashboard.stats is not None and agent.dashboard.stats.llm_calls == 1
    assert agent.dashboard.activity is not None and len(agent.dashboard.activity.decisions) == 1

    await agent.switch_mode(True)
    await agent.refresh_stats()
    stats, activity = agent.dashboard.stats, agent.dashboard.activity
    assert stats is not None and activity is not None
    assert (stats.mode, stats.llm_calls, stats.bets_made, stats.total_pnl) == ("live", 0, 0, 0.0)
    assert (activity.decisions, activity.orders, activity.positions) == ([], [], [])
    assert stats.llm_spend_today_all_modes == pytest.approx(CALL_USAGE.cost_usd)  # budget spans both modes

    advance(agent, clock, timedelta(seconds=61))  # the per-contract LLM cooldown spans modes
    await agent.handle_candidate(yes_candidate(agent))  # live
    await agent.refresh_stats()
    assert agent.dashboard.stats is not None and agent.dashboard.activity is not None
    assert agent.dashboard.stats.llm_calls == 1 and agent.dashboard.stats.bets_made == 1
    assert [d.decision for d in agent.dashboard.activity.decisions] == ["trade"]
    assert [p.ticker for p in agent.dashboard.activity.positions] == [TICKER]


# ---------------------------------------------------------------- research data & hourly evaluation


async def test_market_snapshots_are_recorded_unconditionally(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator())
    await agent.restore()
    await agent.sync_markets()  # no candidate, no LLM call: still sampled
    assert await agent.write_snapshots() == 1
    async with db.session() as s:
        [snap] = (await s.exec(select(MarketSnapshot))).all()
    assert (snap.ticker, snap.yes_bid, snap.yes_ask, snap.seconds_remaining) == (TICKER, YES_BID, YES_ASK, 600)
    assert 0 < snap.model_probability < 1 and snap.historical_volatility > 0
    # Both pricing models are recorded side by side; the agent trades the spot model by default.
    assert snap.spot_model_probability == snap.model_probability
    assert snap.settlement_model_probability is not None and 0 < snap.settlement_model_probability < 1

    advance(agent, clock, timedelta(seconds=1))
    await agent.sync_markets()
    advance(agent, clock, timedelta(seconds=30))  # sample went stale before the next write
    assert await agent.write_snapshots() == 0


async def test_outcomes_are_recorded_for_contracts_we_never_traded(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator())
    await agent.restore()
    await agent.sync_markets()
    await agent.write_snapshots()
    assert await agent.record_contract_outcomes() == 0  # still trading

    clock.now = T0 + timedelta(minutes=11)
    assert isinstance(agent.client, SettlingClient)
    agent.client.results[TICKER] = KalshiMarket(ticker=TICKER, strike_price=STRIKE, close_time=T0 + timedelta(minutes=10), result="no")
    assert await agent.record_contract_outcomes() == 1
    assert await agent.record_contract_outcomes() == 0  # recorded once
    async with db.session() as s:
        outcome = await s.get(ContractOutcome, TICKER)
    assert outcome is not None and outcome.settled_yes is False
    assert await trade_rows(db) == []


async def test_hourly_evaluation_runs_in_a_subprocess_and_reaches_the_dashboard(tmp_path) -> None:
    import httpx

    from web.app import create_app

    url = f"sqlite+aiosqlite:///{tmp_path / 'agent.db'}"
    database = Database(url)
    await database.init()
    clock = Clock(T0)
    agent = make_agent(database, clock, FakeEvaluator(), database_url=url)
    await agent.restore()
    try:
        assert await agent.run_evaluation()
        assert agent.dashboard.evaluation is not None
        assert agent.dashboard.evaluation.backtest.status.startswith("waiting for data")
        async with database.session() as s:
            assert len((await s.exec(select(EvaluationReportRecord))).all()) == 1

        app = create_app(agent.dashboard, database.engine)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
            page = await http.get("/partials/evaluation")
        assert page.status_code == 200 and "Held-out backtest" in page.text and "waiting for data" in page.text
        assert "Trading on: <strong>spot model</strong>" in page.text and "Resting-order experiment" in page.text
    finally:
        await database.dispose()


async def test_no_pricing_or_snapshots_until_volatility_has_a_full_window(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator())
    cold = BtcSpotFeed()
    for i in range(60):  # one minute of history against a 300s lookback
        cold.buffer.add_tick(BtcSpotTick(source="coinbase", price=82_700.0 + (i % 3), timestamp=T0 - timedelta(seconds=59 - i)))
    agent.feed = cold
    agent.state_manager._spot = cold
    await agent.restore()
    assert not agent.spot_warm() and agent.spot_context() is None
    assert agent.scanner.evaluate(TICKER) == []
    await agent.sync_markets()
    assert await agent.write_snapshots() == 0

    for i in range(1, 200):  # now 259s >= 80% of 300s
        cold.buffer.add_tick(BtcSpotTick(source="coinbase", price=82_700.0 + (i % 5), timestamp=T0 + timedelta(seconds=i)))
    clock.now = T0 + timedelta(seconds=199)
    assert agent.spot_warm() and agent.spot_context() is not None


# ---------------------------------------------------------------- pricing model selection & experiments

from services.evaluation import EvaluationReport, HoldoutBacktest  # noqa: E402


def _report(model_contracts: int, best_model: str, weights: dict[str, float]) -> EvaluationReport:
    return EvaluationReport(
        created_at=T0, contracts=model_contracts, snapshots=model_contracts * 100, scores=[],
        best_blend_weight=weights.get(best_model), best_model=best_model, model_contracts=model_contracts,
        blend_weights=weights, calibration=[], horizons=[], edges=[], llm=None,
        backtest=HoldoutBacktest(status="waiting"), notes=[],
    )


async def test_agent_follows_the_evaluation_pricing_choice_only_with_enough_evidence(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator(), model_min_contracts=200)
    await agent.restore()
    candidate = yes_candidate(agent)
    raw = candidate.fair_value.model_probability

    agent.adopt_evaluation(_report(50, "settlement", {"spot": 0.6, "settlement": 0.4}))
    assert (agent.pricing_model, agent.blend_weight) == ("spot", 1.0)  # not enough contracts yet
    assert "50 so far" in agent.dashboard.pricing["reason"]

    agent.adopt_evaluation(_report(250, "settlement", {"spot": 0.6, "settlement": 0.4}))
    assert (agent.pricing_model, agent.blend_weight) == ("settlement", 0.4)
    blended = yes_candidate(agent).fair_value
    mid = (YES_BID + YES_ASK) / 200
    settle = agent.model_values(candidate.market_state, candidate.quote)["settlement"].model_probability
    assert blended.model_probability == pytest.approx(0.4 * settle + 0.6 * mid)
    assert blended.model_probability < raw  # shrunk toward the 78c mid
    assert agent.dashboard.pricing == {
        "model": "settlement", "blend_weight": 0.4, "reason": "hourly evaluation over 250 settled contracts",
        "setting_model": "auto", "setting_blend": "auto",
    }


async def test_fixed_pricing_settings_override_the_evaluation(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator(), pricing_model="spot", market_blend="off")
    await agent.restore()
    agent.adopt_evaluation(_report(500, "settlement", {"spot": 0.3, "settlement": 0.2}))
    assert (agent.pricing_model, agent.blend_weight) == ("spot", 1.0)


async def test_llm_reviewed_candidates_record_a_replayable_input_and_a_maker_shadow(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator())
    await agent.restore()
    assert agent.fee_type == "quadratic"
    await agent.handle_candidate(yes_candidate(agent))

    async with db.session() as s:
        [trace] = (await s.exec(select(EvaluationTraceRecord))).all()
        [shadow] = (await s.exec(select(MakerShadowOrder))).all()
    payload = json.loads(trace.llm_input_json or "")
    assert payload["candidate"]["ticker"] == TICKER and payload["context"]["proposed_side"] == "yes"
    # YES book 76/80: improve the bid by a cent and rest for maker_ttl_sec; no maker fee on this series.
    assert (shadow.side, shadow.limit_cents, shadow.joined_queue, shadow.taker_ask_cents) == ("yes", 77, False, YES_ASK)
    assert shadow.maker_fee_cents == 0 and shadow.llm_should_bet is True
    assert shadow.cancel_at == T0 + timedelta(seconds=180) and shadow.filled is None

    assert await agent.resolve_maker_shadows() == 0  # still resting
    assert isinstance(agent.client, SettlingClient)
    agent.client.trades = [
        # A taker buying YES lifts asks: never fills our YES bid.
        PublicTrade(ticker=TICKER, created_time=T0 + timedelta(seconds=20), yes_price_cents=80, no_price_cents=20, count=5, taker_side="yes"),
        # A taker buying NO at 24c sells YES at 76c, below our 77c bid: we'd have been hit first.
        PublicTrade(ticker=TICKER, created_time=T0 + timedelta(seconds=90), yes_price_cents=76, no_price_cents=24, count=3, taker_side="no"),
    ]
    clock.now = T0 + timedelta(seconds=181)
    assert await agent.resolve_maker_shadows() == 1
    async with db.session() as s:
        [shadow] = (await s.exec(select(MakerShadowOrder))).all()
    assert shadow.filled is True and shadow.filled_at == T0 + timedelta(seconds=90) and shadow.fill_trade_cents == 76
    assert await agent.resolve_maker_shadows() == 0


async def test_vetoed_candidates_get_a_maker_shadow_too_but_unpaid_screens_do_not(db: Database) -> None:
    clock = Clock(T0)
    agent = make_agent(db, clock, FakeEvaluator(should_bet=False))
    await agent.restore()
    await agent.handle_candidate(yes_candidate(agent))
    agent_no_llm = make_agent(db, clock, None)
    await agent_no_llm.restore()
    clock.now += timedelta(seconds=1)
    await agent_no_llm.handle_candidate(yes_candidate(agent_no_llm))
    async with db.session() as s:
        shadows = (await s.exec(select(MakerShadowOrder))).all()
        traces = (await s.exec(select(EvaluationTraceRecord))).all()
    assert [sh.llm_should_bet for sh in shadows] == [False]
    assert len(traces) == 2 and traces[1].llm_input_json is None  # no LLM call, nothing to replay
