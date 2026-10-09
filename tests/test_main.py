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
