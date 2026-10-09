"""Hourly evaluation: probability scoring, realized edge, LLM value, held-out backtest, and the job."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sqlmodel import select

from core.schemas import StrategyParameters
from db.database import Database
from db.models import (
    ContractOutcome,
    EvaluationReportRecord,
    EvaluationTraceRecord,
    MarketSnapshot,
    StrategyParametersRecord,
    TradeRecord,
)
from services.evaluation import (
    brier,
    build_report,
    calibration,
    fit_blend_weight,
    holdout_backtest,
    llm_value,
    log_loss,
    realized_edge,
    run_once,
)
from services.risk_engine import kalshi_taker_fee_cents
from tests.test_backtest_and_ui import SMALL_GRID, synthetic_history

NOW = datetime(2026, 10, 9, 22, 0, tzinfo=timezone.utc)


def test_scores_reward_better_probabilities() -> None:
    y = np.array([1.0, 0.0, 1.0, 1.0])
    sharp, coin = np.array([0.9, 0.1, 0.8, 0.9]), np.full(4, 0.5)
    assert brier(coin, y) == pytest.approx(0.25)
    assert brier(sharp, y) < brier(coin, y) and log_loss(sharp, y) < log_loss(coin, y)
    assert np.isfinite(log_loss(np.array([0.0, 1.0]), np.array([1.0, 0.0])))  # clipped, not infinite


def test_blend_weight_finds_the_informative_predictor() -> None:
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=truth.size) < truth).astype(float)
    noise = np.clip(truth + rng.normal(0, 0.3, truth.size), 0.01, 0.99)
    assert fit_blend_weight(truth, noise, y) >= 0.8  # model knows the truth, market is noise
    assert fit_blend_weight(noise, truth, y) <= 0.2  # market knows the truth


def test_calibration_buckets() -> None:
    p = np.array([0.05, 0.08, 0.55, 0.58, 0.95])
    y = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
    buckets = {round(b.lo, 1): b for b in calibration(p, y)}
    assert buckets[0.0].n == 2 and buckets[0.0].realized == 0.0
    assert buckets[0.5].n == 2 and buckets[0.5].realized == 0.5
    assert buckets[0.9].n == 1


def _trade(side: str, price: float, count: int, fees: int, settled_yes: bool, *, is_paper: bool = True) -> TradeRecord:
    won = settled_yes == (side == "yes")
    pnl = (100 * count if won else 0) - round(price * count) - fees
    return TradeRecord(ticker="KXBTC15M-X", side=side, avg_fill_price_cents=price, filled_count=count, fees_cents=fees,
                       settled_yes=settled_yes, realized_pnl_cents=pnl, is_paper=is_paper, entry_time=NOW)


def test_realized_edge_per_contract_with_interval() -> None:
    trades = [_trade("yes", 50, 10, 18, True), _trade("no", 40, 5, 9, True), _trade("yes", 60, 2, 4, True)]
    report = realized_edge(trades, "paper")
    assert report is not None and report.trades == 3
    per_contract = [100 - 50 - 1.8, 0 - 40 - 1.8, 100 - 60 - 2.0]
    assert report.edge_cents == pytest.approx(np.mean(per_contract))
    assert report.ci_low_cents < report.edge_cents < report.ci_high_cents
    assert report.win_rate == pytest.approx(2 / 3)
    assert realized_edge([], "live") is None


def test_llm_value_counts_what_vetoes_saved() -> None:
    def trace(side: str, ask: int, approved: bool, p_llm: float) -> EvaluationTraceRecord:
        return EvaluationTraceRecord(ticker="T", proposed_side=side, proposed_ask_cents=ask, llm_should_bet=approved,
                                     llm_adjusted_probability=p_llm, model_probability=0.6)

    rows = [
        (trace("yes", 50, True, 0.7), True),  # approved, won
        (trace("yes", 50, False, 0.3), False),  # vetoed, would have lost
        (trace("no", 30, False, 0.9), True),  # vetoed, would have lost
    ]
    value = llm_value(rows)
    assert value is not None and (value.approved, value.vetoed) == (1, 2)
    lost = [-50 - kalshi_taker_fee_cents(1, 50), -30 - kalshi_taker_fee_cents(1, 30)]
    assert value.saved_cents == pytest.approx(-sum(lost))
    assert value.approved_edge_cents == pytest.approx(50 - kalshi_taker_fee_cents(1, 50))


def test_holdout_backtest_waits_for_enough_data_then_compares_on_unseen_trades() -> None:
    ticks, trades = synthetic_history(days=6)
    waiting, none = holdout_backtest(ticks, trades, StrategyParameters(), grid=SMALL_GRID, min_trades=10_000)
    assert none is None and waiting.status.startswith("waiting for data")

    report, _ = holdout_backtest(ticks, trades, StrategyParameters(), grid=SMALL_GRID, min_trades=50)
    assert report.status == "ok" and report.current is not None and report.best is not None
    assert report.train_trades > report.test_trades > 0


@pytest.fixture
async def db():
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.init()
    yield database
    await database.dispose()


async def test_report_scores_model_against_market_on_settled_snapshots(db: Database) -> None:
    async with db.session() as s:
        for i, (ticker, yes) in enumerate([("KXBTC15M-A", True), ("KXBTC15M-B", False)]):
            close = NOW - timedelta(minutes=30 - 15 * i)
            s.add(ContractOutcome(ticker=ticker, strike_price=82_500, close_time=close, settled_yes=yes))
            for k in range(10):
                s.add(MarketSnapshot(
                    ts=close - timedelta(seconds=600 - 30 * k), ticker=ticker, strike_price=82_500, close_time=close,
                    seconds_remaining=600 - 30 * k, yes_bid=48, yes_ask=52, spot_price=82_500, historical_volatility=0.3,
                    momentum_1m=0.0, order_book_imbalance=0.0, model_probability=0.8 if yes else 0.2,
                ))
        s.add(_trade("yes", 50, 1, 2, True))
        await s.commit()

    report = await build_report(db, NOW, apply_params=True, cooldown_hours=24, min_trades=300)
    assert (report.contracts, report.snapshots) == (2, 20)
    by_name = {sc.name.split(" (")[0]: sc for sc in report.scores}
    assert by_name["Our model"].brier < by_name["Market"].brier  # the model was right here
    assert report.best_blend_weight == pytest.approx(1.0)
    assert [e.mode for e in report.edges] == ["paper"]
    assert report.backtest.status.startswith("waiting for data") and not report.backtest.applied


async def test_run_once_persists_the_report(tmp_path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'eval.db'}"
    report = await run_once(url, min_trades=300)
    database = Database(url)
    async with database.session() as s:
        [row] = (await s.exec(select(EvaluationReportRecord))).all()
        assert (await s.exec(select(StrategyParametersRecord))).all() == []  # nothing applied
    await database.dispose()
    assert row.report_json == report.model_dump_json()


async def test_evaluation_panel_renders_a_full_report(db: Database) -> None:
    import httpx

    from web.app import DashboardState, create_app

    async with db.session() as s:
        close = NOW - timedelta(minutes=20)
        s.add(ContractOutcome(ticker="KXBTC15M-A", strike_price=82_500, close_time=close, settled_yes=True))
        for k in range(5):
            s.add(MarketSnapshot(ts=close - timedelta(seconds=300 - 60 * k), ticker="KXBTC15M-A", strike_price=82_500,
                                 close_time=close, seconds_remaining=300 - 60 * k, yes_bid=58, yes_ask=60, spot_price=82_600,
                                 historical_volatility=0.3, momentum_1m=0.0, order_book_imbalance=0.1, model_probability=0.7))
        s.add(EvaluationTraceRecord(ticker="KXBTC15M-A", strike_price=82_500, expiration_time=close, seconds_remaining=300,
                                    current_spot_price=82_600, historical_volatility=0.3, momentum_1m=0.0, order_book_imbalance=0.1,
                                    model_probability=0.7, fair_cents=70, gross_edge=0.1, volatility_used=0.3, d2_score=0.5,
                                    proposed_side="yes", proposed_ask_cents=60, llm_should_bet=False, llm_adjusted_probability=0.55,
                                    decision="skip", created_at=close - timedelta(minutes=5)))
        s.add(_trade("yes", 60, 1, 2, True))
        s.add(_trade("no", 41, 1, 2, True, is_paper=False))
        await s.commit()
    report = await build_report(db, NOW, apply_params=False, cooldown_hours=24, min_trades=300)
    ticks, trades = synthetic_history(days=6)
    report.backtest, _ = holdout_backtest(ticks, trades, StrategyParameters(), grid=SMALL_GRID, min_trades=50)

    state = DashboardState()
    state.evaluation = report
    app = create_app(state, db.engine)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
        page = await http.get("/partials/evaluation")
    assert page.status_code == 200
    for text in ("Our model", "Market (mid price)", "best", "PAPER", "LIVE", "Vetoed", "Active parameters", "Best tuned"):
        assert text in page.text, text
