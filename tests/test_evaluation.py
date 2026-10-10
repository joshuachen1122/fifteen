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
    assert by_name["Spot model"].brier < by_name["Market"].brier  # the model was right here
    assert report.best_model == "spot" and report.model_contracts == 2
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
    for text in ("Spot model", "Market (mid price)", "best", "PAPER", "LIVE", "Vetoed", "Active parameters", "Best tuned"):
        assert text in page.text, text


# ---------------------------------------------------------------- model choice, learned model, experiments

import json  # noqa: E402

from core.schemas import FairValueOutput, LLMVerdict  # noqa: E402
from db.models import LLMReplayResult, MakerShadowOrder  # noqa: E402
from services.evaluation import (  # noqa: E402
    fit_logistic,
    learned_features,
    llm_replay_summaries,
    maker_reports,
    predict_logistic,
    walk_forward_learned,
)
from services.llm_evaluator import LLMUsage  # noqa: E402
from services.llm_replay import run_replays  # noqa: E402


async def test_report_prefers_the_settlement_model_when_it_predicts_better(db: Database) -> None:
    async with db.session() as s:
        for i in range(6):
            ticker, yes = f"KXBTC15M-{i}", i % 2 == 0
            close = NOW - timedelta(minutes=90 - 15 * i)
            s.add(ContractOutcome(ticker=ticker, strike_price=82_500, close_time=close, settled_yes=yes))
            for k in range(10):
                s.add(MarketSnapshot(
                    ts=close - timedelta(seconds=300 - 30 * k), ticker=ticker, strike_price=82_500, close_time=close,
                    seconds_remaining=300 - 30 * k, yes_bid=48, yes_ask=52, spot_price=82_500, historical_volatility=0.3,
                    momentum_1m=0.0, order_book_imbalance=0.0, model_probability=0.6 if yes else 0.4,
                    spot_model_probability=0.6 if yes else 0.4, settlement_model_probability=0.75 if yes else 0.25,
                ))
        # Pre-release rows (spot only) are left out of the head-to-head comparison.
        s.add(ContractOutcome(ticker="OLD", strike_price=1, close_time=NOW - timedelta(days=1), settled_yes=False))
        s.add(MarketSnapshot(
            ts=NOW - timedelta(days=1), ticker="OLD", strike_price=1, close_time=NOW - timedelta(days=1), seconds_remaining=60,
            yes_bid=1, yes_ask=3, spot_price=1, historical_volatility=0.3, momentum_1m=0, order_book_imbalance=0,
            model_probability=0.99,
        ))
        await s.commit()
    report = await build_report(db, NOW, apply_params=False, cooldown_hours=24, min_trades=300)
    assert report.best_model == "settlement" and report.model_contracts == 6 and report.contracts == 7
    assert set(report.blend_weights) == {"spot", "settlement"} and report.best_blend_weight == report.blend_weights["settlement"]
    names = [sc.name for sc in report.scores]
    assert names[:3] == ["Market (mid price)", "Spot model (price at close)", "Settlement model (60s average)"]
    assert all(h.brier_settlement is not None and h.brier_settlement < h.brier_model for h in report.horizons)
    assert report.learned is not None and report.learned.status.startswith("waiting for data: 6 of 100")


def test_walk_forward_learned_model_beats_a_noisy_market_out_of_sample() -> None:
    rng = np.random.default_rng(3)
    contracts, per = 160, 20
    truth = np.repeat(rng.uniform(0.05, 0.95, contracts), per)
    y = np.repeat((rng.uniform(size=contracts) < truth[::per]).astype(float), per)
    p_model = np.clip(truth, 0.01, 0.99)  # the model knows the truth...
    p_market = np.clip(truth + rng.normal(0, 0.25, truth.size), 0.01, 0.99)  # ...the market is noisy
    seconds = np.tile(np.linspace(900, 30, per), contracts)
    x = learned_features(p_market, p_model, seconds, np.zeros_like(truth), np.full_like(truth, 0.3), np.zeros_like(truth))
    order = np.repeat(np.arange(contracts), per)
    report = walk_forward_learned(order, x, y, p_market, p_model)
    assert report.status == "ok" and report.test_contracts == 80 and report.test_snapshots == 80 * per
    assert report.log_loss_learned < report.log_loss_market
    assert report.coefficients["logit_model"] > report.coefficients["logit_market"]
    assert walk_forward_learned(order[:200], x[:200], y[:200], p_market[:200], p_model[:200]).status.startswith("waiting")


def test_logistic_fit_recovers_a_known_relationship() -> None:
    rng = np.random.default_rng(5)
    x = np.column_stack([np.ones(20_000), rng.normal(size=20_000)])
    y = (rng.uniform(size=20_000) < 1 / (1 + np.exp(-(0.5 + 2.0 * x[:, 1])))).astype(float)
    beta = fit_logistic(x, y, l2=0.0)
    assert beta == pytest.approx([0.5, 2.0], abs=0.1)
    assert predict_logistic(x[:1], beta).shape == (1,)


def _shadow(side: str, limit: int, ask: int, filled: bool, approved: bool = True) -> MakerShadowOrder:
    return MakerShadowOrder(
        ticker="T", side=side, is_paper=True, llm_should_bet=approved, close_time=NOW, cancel_at=NOW, limit_cents=limit,
        joined_queue=False, taker_ask_cents=ask, maker_fee_cents=0, win_probability=0.6, filled=filled,
    )


def test_maker_report_compares_resting_bids_with_buying_at_the_ask() -> None:
    rows = [
        (_shadow("yes", 50, 53, True), True),  # filled, won: +50 vs taker +47 - fee
        (_shadow("yes", 50, 53, True), False),  # filled, lost: -50 (adverse selection)
        (_shadow("no", 30, 33, False), False),  # not filled; buying NO at the ask would have won
        (_shadow("yes", 40, 43, False, approved=False), False),
    ]
    all_, approved = maker_reports(rows)
    assert (all_.group, all_.orders, all_.filled, all_.fill_rate) == ("All LLM-reviewed", 4, 2, 0.5)
    taker_fee = kalshi_taker_fee_cents(1, 53)
    assert all_.maker_edge_filled_cents == pytest.approx(0.0)
    assert all_.taker_edge_filled_cents == pytest.approx(((47 - taker_fee) + (-53 - taker_fee)) / 2)
    assert all_.win_rate_filled == 0.5 and all_.win_rate_unfilled == 0.5
    assert all_.maker_per_opportunity_cents == pytest.approx(0.0)
    assert approved.orders == 3 and approved.filled == 2
    assert maker_reports([]) == []


def _trace(i: int, approved: bool, p: float, ask: int = 50) -> EvaluationTraceRecord:
    return EvaluationTraceRecord(
        id=i, ticker=f"T{i}", strike_price=1, expiration_time=NOW, seconds_remaining=300, current_spot_price=1,
        historical_volatility=0.3, momentum_1m=0, order_book_imbalance=0, proposed_side="yes", proposed_ask_cents=ask,
        model_probability=0.6, fair_cents=60, gross_edge=0.1, volatility_used=0.3, d2_score=0.2,
        llm_should_bet=approved, llm_adjusted_probability=p, llm_model="claude-sonnet-5-5", llm_cost_usd=0.004,
        decision="trade" if approved else "skip",
    )


def test_llm_replay_summary_scores_both_models_on_the_same_decisions() -> None:
    def replay(i: int, approve: bool, p: float, failed: bool = False) -> LLMReplayResult:
        return LLMReplayResult(trace_id=i, model="claude-haiku-5-5", effort="low", should_bet=approve,
                               adjusted_probability=p, failed=failed, cost_usd=0.0003)

    rows = [
        (_trace(1, True, 0.7), replay(1, True, 0.8), True),
        (_trace(2, False, 0.4), replay(2, True, 0.6), False),  # Haiku would have approved a loser
        (_trace(3, True, 0.6), replay(3, False, 0.5, failed=True), True),  # failed replays don't count
    ]
    [summary] = llm_replay_summaries(rows)
    assert (summary.model, summary.live_model, summary.decisions, summary.failures) == (
        "claude-haiku-5-5", "claude-sonnet-5-5", 3, 1)
    assert summary.agreement == 0.5 and (summary.live_approved, summary.replay_approved) == (1, 2)
    win, loss = 50 - kalshi_taker_fee_cents(1, 50), -50 - kalshi_taker_fee_cents(1, 50)
    assert summary.live_approved_edge_cents == pytest.approx(win)
    assert summary.replay_approved_edge_cents == pytest.approx((win + loss) / 2)
    assert summary.cost_per_call_replay == pytest.approx(0.0003) and summary.cost_per_call_live == pytest.approx(0.004)


class _ReplayEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[FairValueOutput, dict]] = []

    async def evaluate_with_usage(self, candidate: FairValueOutput, context: dict):
        self.calls.append((candidate, context))
        verdict = LLMVerdict(should_bet=False, adjusted_probability=0.5, confidence_score=0.5, reasoning="r", risk_signals=[])
        return verdict, LLMUsage(model="claude-haiku-5-5", input_tokens=1_000_000)  # $0.10 per call


async def test_replays_only_settled_logged_decisions_within_the_daily_budget(db: Database) -> None:
    candidate = FairValueOutput(ticker="T1", strike_price=1, model_probability=0.6, fair_cents=60, gross_edge=0.1,
                                volatility_used=0.3, d2_score=0.2)
    async with db.session() as s:
        for i in (1, 2, 3, 4):
            trace = _trace(i, True, 0.6)
            trace.created_at = NOW - timedelta(minutes=10 * (5 - i))
            if i != 4:  # trace 4 predates input logging
                trace.llm_input_json = json.dumps({"candidate": candidate.model_dump(mode="json"), "context": {"i": i}})
            s.add(trace)
            if i != 3:  # contract 3 hasn't settled
                s.add(ContractOutcome(ticker=f"T{i}", strike_price=1, close_time=NOW, settled_yes=True))
        await s.commit()

    fake = _ReplayEvaluator()
    factory = lambda model, effort: fake  # noqa: E731
    n = await run_replays(db, model="claude-haiku-5-5", effort="low", daily_budget_usd=0.15, now=NOW, evaluator_factory=factory)
    assert n == 2 and [c[1]["i"] for c in fake.calls] == [2, 1]  # newest first; budget hit after the 2nd
    assert await run_replays(db, model="claude-haiku-5-5", effort="low", daily_budget_usd=0.15, now=NOW, evaluator_factory=factory) == 0
    async with db.session() as s:
        rows = (await s.exec(select(LLMReplayResult))).all()
    assert sorted(r.trace_id for r in rows) == [1, 2] and all(not r.failed for r in rows)
    assert await run_replays(db, model="claude-haiku-5-5", effort="low", daily_budget_usd=0, now=NOW) == 0


async def test_models_are_judged_on_calibrated_snapshots_once_they_are_enough_evidence(db: Database) -> None:
    async with db.session() as s:
        for i in range(8):
            ticker, yes, calibrated = f"KXBTC15M-{i}", i % 2 == 0, i >= 4
            close = NOW - timedelta(minutes=150 - 15 * i)
            s.add(ContractOutcome(ticker=ticker, strike_price=82_500, close_time=close, settled_yes=yes))
            # Uncalibrated generation: settlement model better. Calibrated generation: spot model better.
            good, bad = (0.8, 0.55) if yes else (0.2, 0.45)
            spot, settle = (good, bad) if calibrated else (bad, good)
            for k in range(10):
                s.add(MarketSnapshot(
                    ts=close - timedelta(seconds=300 - 30 * k), ticker=ticker, strike_price=82_500, close_time=close,
                    seconds_remaining=300 - 30 * k, yes_bid=48, yes_ask=52, spot_price=82_504, historical_volatility=0.3,
                    momentum_1m=0.0, order_book_imbalance=0.0, model_probability=spot, spot_model_probability=spot,
                    settlement_model_probability=settle, index_gap=4.0 if calibrated else None,
                ))
        await s.commit()
    mixed = await build_report(db, NOW, apply_params=False, cooldown_hours=24, min_trades=300, min_model_contracts=5)
    assert mixed.model_contracts == 8 and "cover 4 settled contracts" in " ".join(mixed.notes)
    scoped = await build_report(db, NOW, apply_params=False, cooldown_hours=24, min_trades=300, min_model_contracts=4)
    assert scoped.model_contracts == 4 and scoped.best_model == "spot"
    assert any("corrections only" in n for n in scoped.notes)
