"""Hourly evaluation of the pricing models, the LLM gate, execution style, and realized edge.

Runs in its own low-priority process (`python -m services.evaluation`) so the CPU-heavy
backtest grid never competes with the live order-book feed. Writes one `EvaluationReport`
row per run; the dashboard shows the latest. New strategy parameters are applied only when a
held-out backtest clearly beats the active ones (and never more than once per cooldown).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import numpy as np
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import col, select

from core.constants import DEFAULT_DATABASE_URL
from core.schemas import StrategyParameters
from db.database import Database
from db.models import (
    BtcPriceHistory,
    ContractOutcome,
    EvaluationReportRecord,
    EvaluationTraceRecord,
    LLMReplayResult,
    MakerShadowOrder,
    MarketSnapshot,
    StrategyParametersRecord,
    TradeRecord,
)
from pricing.btc_digital_option import MOMENTUM_Z_CLIP, SECONDS_PER_YEAR
from services.backtester import Backtester, BacktestResult
from services.llm_replay import run_replays
from services.optimizer import Optimizer, ParameterGrid, load_active_parameters
from services.risk_engine import kalshi_taker_fee_cents

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 30
PROB_CLIP = 0.005  # keeps log loss finite when the market quotes 0c/100c
HORIZONS = [(0, 60, "< 1 min"), (60, 180, "1-3 min"), (180, 420, "3-7 min"), (420, 10_000, "7-15 min")]
MODEL_LABELS = {"spot": "Spot model (price at close)", "settlement": "Settlement model (60s average)"}
LEARNED_MIN_CONTRACTS = 100  # walk-forward needs enough contracts to train on before the first test fold
LEARNED_FOLDS = 5
LEARNED_MAX_ROWS = 200_000
MAX_SNAPSHOT_ROWS = 150_000  # bound the job's memory on the 1 GB VM: sample snapshots evenly beyond this  # subsample very large training sets (snapshots are highly correlated anyway)


class ProbabilityScore(BaseModel):
    name: str
    brier: float  # lower is better; 0.25 = coin flip at 50%
    log_loss: float
    n: int


class CalibrationBucket(BaseModel):
    lo: float
    hi: float
    n: int
    mean_predicted: float
    realized: float


class HorizonScore(BaseModel):
    label: str
    n: int
    brier_market: float
    brier_model: float  # spot model
    brier_settlement: Optional[float] = None


class EdgeReport(BaseModel):
    mode: str
    trades: int
    win_rate: float
    avg_price_cents: float
    avg_fee_cents: float  # per contract
    edge_cents: float  # mean realized edge per contract after fees
    ci_low_cents: float  # 95% interval for the mean
    ci_high_cents: float
    total_pnl: float  # USD


class LLMValue(BaseModel):
    """Hypothetical 1-contract result of every LLM-reviewed candidate, by verdict."""

    approved: int
    vetoed: int
    approved_edge_cents: Optional[float]
    vetoed_edge_cents: Optional[float]
    saved_cents: float  # minus the hypothetical PnL of everything it vetoed (positive = vetoes helped)
    brier_llm: Optional[float]
    brier_model: Optional[float]


class LearnedModelReport(BaseModel):
    """Walk-forward logistic model on snapshot features; scored only on contracts it never saw."""

    status: str
    test_contracts: int = 0
    test_snapshots: int = 0
    log_loss_learned: Optional[float] = None
    log_loss_market: Optional[float] = None
    log_loss_model: Optional[float] = None  # best hand-built model on the same rows
    brier_learned: Optional[float] = None
    brier_market: Optional[float] = None
    coefficients: dict[str, float] = {}


class MakerReport(BaseModel):
    """Shadow resting bids vs buying at the ask, per contract (cents)."""

    group: str
    orders: int
    filled: int
    fill_rate: float
    maker_edge_filled_cents: Optional[float]  # result per contract of the bids that filled
    taker_edge_filled_cents: Optional[float]  # buying at the ask on those same candidates
    taker_edge_all_cents: float  # buying at the ask on every candidate
    win_rate_filled: Optional[float]
    win_rate_unfilled: Optional[float]
    maker_per_opportunity_cents: float  # unfilled bids count as 0
    taker_per_opportunity_cents: float


class LLMReplaySummary(BaseModel):
    """A cheaper model re-asked on the same logged decisions as the live model."""

    model: str
    live_model: str
    decisions: int
    failures: int
    agreement: float
    live_approved: int
    replay_approved: int
    live_approved_edge_cents: Optional[float]
    replay_approved_edge_cents: Optional[float]
    brier_live: Optional[float]
    brier_replay: Optional[float]
    cost_per_call_live: Optional[float]
    cost_per_call_replay: Optional[float]


class StrategySummary(BaseModel):
    params: StrategyParameters
    trades: int
    total_net_pnl: float
    sharpe_ratio: float
    win_rate: float


class HoldoutBacktest(BaseModel):
    status: str
    train_trades: int = 0
    test_trades: int = 0
    current: Optional[StrategySummary] = None
    best: Optional[StrategySummary] = None
    applied: bool = False


class EvaluationReport(BaseModel):
    created_at: datetime
    contracts: int
    snapshots: int
    scores: list[ProbabilityScore]
    best_blend_weight: Optional[float]  # weight on the best model in w*model + (1-w)*market
    best_model: Optional[str] = None  # "spot" | "settlement": lower log loss on the comparison rows
    model_contracts: int = 0  # settled contracts behind the model comparison
    blend_weights: dict[str, float] = {}  # fitted market-blend weight per model
    calibration: list[CalibrationBucket]
    horizons: list[HorizonScore]
    edges: list[EdgeReport]
    llm: Optional[LLMValue]
    backtest: HoldoutBacktest
    notes: list[str]
    learned: Optional[LearnedModelReport] = None
    maker: list[MakerReport] = []
    llm_replays: list[LLMReplaySummary] = []


# ----------------------------------------------------------------------------- metrics


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, PROB_CLIP, 1 - PROB_CLIP)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def score(name: str, p: np.ndarray, y: np.ndarray) -> ProbabilityScore:
    return ProbabilityScore(name=name, brier=brier(p, y), log_loss=log_loss(p, y), n=len(y))


def fit_blend_weight(p_model: np.ndarray, p_market: np.ndarray, y: np.ndarray) -> float:
    """Weight w in [0, 1] on the model that minimizes log loss of w*model + (1-w)*market."""
    weights = np.linspace(0.0, 1.0, 21)
    losses = [log_loss(w * p_model + (1 - w) * p_market, y) for w in weights]
    return float(weights[int(np.argmin(losses))])


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[CalibrationBucket]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if mask.any():
            buckets.append(
                CalibrationBucket(
                    lo=float(lo), hi=float(hi), n=int(mask.sum()),
                    mean_predicted=float(p[mask].mean()), realized=float(y[mask].mean()),
                )
            )
    return buckets


def horizon_scores(
    seconds: np.ndarray,
    p_model: np.ndarray,
    p_market: np.ndarray,
    y: np.ndarray,
    p_settlement: Optional[np.ndarray] = None,
) -> list[HorizonScore]:
    out = []
    for lo, hi, label in HORIZONS:
        mask = (seconds >= lo) & (seconds < hi)
        if mask.any():
            out.append(
                HorizonScore(
                    label=label, n=int(mask.sum()),
                    brier_market=brier(p_market[mask], y[mask]), brier_model=brier(p_model[mask], y[mask]),
                    brier_settlement=None if p_settlement is None else brier(p_settlement[mask], y[mask]),
                )
            )
    return out


# ----------------------------------------------------------------------------- learned model

LEARNED_FEATURES = [
    "intercept", "logit_market", "logit_model", "logit_market_x_time", "logit_model_x_time",
    "momentum_z", "order_book_imbalance",
]


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1 - p))


def learned_features(
    p_market: np.ndarray, p_model: np.ndarray, seconds: np.ndarray, momentum_pct: np.ndarray,
    vol: np.ndarray, obi: np.ndarray,
) -> np.ndarray:
    """Design matrix: the market and model views (and how much each matters as time runs out)
    plus the microstructure signals the hand-built model folds in with fixed weights."""
    t = np.clip(seconds, 0, 900) / 900.0
    lm, lp = _logit(p_market), _logit(p_model)
    sigma_1m = vol * math.sqrt(60.0 / SECONDS_PER_YEAR)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma_1m > 0, (momentum_pct / 100.0) / sigma_1m, 0.0)
    z = np.clip(np.nan_to_num(z), -MOMENTUM_Z_CLIP, MOMENTUM_Z_CLIP)
    return np.column_stack([np.ones_like(t), lm, lp, lm * t, lp * t, z, np.clip(obi, -1, 1)])


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iterations: int = 50) -> np.ndarray:
    """L2-regularized logistic regression by Newton's method (the intercept is not penalized)."""
    beta = np.zeros(x.shape[1])
    penalty = np.full(x.shape[1], l2)
    penalty[0] = 0.0
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
        grad = x.T @ (p - y) + penalty * beta
        w = p * (1 - p)
        hess = (x * w[:, None]).T @ x + np.diag(penalty) + 1e-9 * np.eye(x.shape[1])
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def predict_logistic(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))


def walk_forward_learned(
    contract_order: np.ndarray, x: np.ndarray, y: np.ndarray, p_market: np.ndarray, p_model: np.ndarray,
    *, folds: int = LEARNED_FOLDS, min_contracts: int = LEARNED_MIN_CONTRACTS,
) -> LearnedModelReport:
    """`contract_order[i]` ranks row i's contract by close time. Each fold trains only on contracts
    that closed before the fold's first contract, then predicts the fold."""
    n_contracts = int(contract_order.max()) + 1 if len(contract_order) else 0
    if n_contracts < min_contracts:
        return LearnedModelReport(status=f"waiting for data: {n_contracts} of {min_contracts} settled contracts")
    first_test = n_contracts // 2  # always train on at least half of the history
    bounds = np.linspace(first_test, n_contracts, folds + 1).astype(int)
    preds = np.full(len(y), np.nan)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        train = contract_order < lo
        test = (contract_order >= lo) & (contract_order < hi)
        if not test.any() or not train.any():
            continue
        idx = np.flatnonzero(train)
        if len(idx) > LEARNED_MAX_ROWS:
            idx = idx[:: math.ceil(len(idx) / LEARNED_MAX_ROWS)]
        beta = fit_logistic(x[idx], y[idx])
        preds[test] = predict_logistic(x[test], beta)
    tested = ~np.isnan(preds)
    final = fit_logistic(x[:: max(1, math.ceil(len(y) / LEARNED_MAX_ROWS))], y[:: max(1, math.ceil(len(y) / LEARNED_MAX_ROWS))])
    return LearnedModelReport(
        status="ok",
        test_contracts=int(len(np.unique(contract_order[tested]))),
        test_snapshots=int(tested.sum()),
        log_loss_learned=log_loss(preds[tested], y[tested]),
        log_loss_market=log_loss(p_market[tested], y[tested]),
        log_loss_model=log_loss(p_model[tested], y[tested]),
        brier_learned=brier(preds[tested], y[tested]),
        brier_market=brier(p_market[tested], y[tested]),
        coefficients={name: round(float(b), 4) for name, b in zip(LEARNED_FEATURES, final)},
    )


# ----------------------------------------------------------------------------- execution & LLM experiments


def maker_reports(rows: Sequence[tuple[MakerShadowOrder, bool]]) -> list[MakerReport]:
    """`rows` pairs each resolved shadow bid with whether its contract settled YES."""
    groups = [("All LLM-reviewed", list(rows)), ("LLM-approved only", [r for r in rows if r[0].llm_should_bet])]
    out = []
    for name, group in groups:
        if not group:
            continue
        won = np.array([1.0 if yes == (o.side == "yes") else 0.0 for o, yes in group])
        filled = np.array([bool(o.filled) for o, _ in group])
        maker = won * 100 - np.array([o.limit_cents + o.maker_fee_cents for o, _ in group], dtype=float)
        taker = won * 100 - np.array(
            [o.taker_ask_cents + kalshi_taker_fee_cents(1, o.taker_ask_cents) for o, _ in group], dtype=float
        )
        out.append(
            MakerReport(
                group=name,
                orders=len(group),
                filled=int(filled.sum()),
                fill_rate=float(filled.mean()),
                maker_edge_filled_cents=float(maker[filled].mean()) if filled.any() else None,
                taker_edge_filled_cents=float(taker[filled].mean()) if filled.any() else None,
                taker_edge_all_cents=float(taker.mean()),
                win_rate_filled=float(won[filled].mean()) if filled.any() else None,
                win_rate_unfilled=float(won[~filled].mean()) if (~filled).any() else None,
                maker_per_opportunity_cents=float(np.where(filled, maker, 0.0).mean()),
                taker_per_opportunity_cents=float(taker.mean()),
            )
        )
    return out


def _hypothetical_cents(trace: EvaluationTraceRecord, settled_yes: bool) -> float:
    ask = trace.proposed_ask_cents or 0
    won = settled_yes == (trace.proposed_side == "yes")
    return (100.0 if won else 0.0) - ask - kalshi_taker_fee_cents(1, ask)


def llm_replay_summaries(rows: Sequence[tuple[EvaluationTraceRecord, LLMReplayResult, bool]]) -> list[LLMReplaySummary]:
    """`rows` pairs each live decision with one replay of it and the contract's outcome."""
    by_model: dict[str, list[tuple[EvaluationTraceRecord, LLMReplayResult, bool]]] = {}
    for row in rows:
        by_model.setdefault(row[1].model, []).append(row)
    out = []
    for model, group in sorted(by_model.items()):
        ok = [r for r in group if not r[1].failed]
        y = np.array([1.0 if yes else 0.0 for _, _, yes in ok])
        live_p = np.array([t.llm_adjusted_probability or t.model_probability for t, _, _ in ok])
        replay_p = np.array([r.adjusted_probability for _, r, _ in ok])
        live_ok = [_hypothetical_cents(t, yes) for t, _, yes in ok if t.llm_should_bet]
        replay_ok = [_hypothetical_cents(t, yes) for t, r, yes in ok if r.should_bet]
        live_costs = [t.llm_cost_usd for t, _, _ in group if t.llm_cost_usd is not None]
        replay_costs = [r.cost_usd for _, r, _ in group if r.cost_usd is not None]
        live_models = {t.llm_model for t, _, _ in group if t.llm_model}
        out.append(
            LLMReplaySummary(
                model=model,
                live_model=", ".join(sorted(live_models)) or "unknown",
                decisions=len(group),
                failures=len(group) - len(ok),
                agreement=float(np.mean([t.llm_should_bet == r.should_bet for t, r, _ in ok])) if ok else 0.0,
                live_approved=len(live_ok),
                replay_approved=len(replay_ok),
                live_approved_edge_cents=float(np.mean(live_ok)) if live_ok else None,
                replay_approved_edge_cents=float(np.mean(replay_ok)) if replay_ok else None,
                brier_live=brier(live_p, y) if ok else None,
                brier_replay=brier(replay_p, y) if ok else None,
                cost_per_call_live=float(np.mean(live_costs)) if live_costs else None,
                cost_per_call_replay=float(np.mean(replay_costs)) if replay_costs else None,
            )
        )
    return out


def realized_edge(trades: Sequence[TradeRecord], mode: str) -> Optional[EdgeReport]:
    """Per-contract edge of settled trades: payout - price - fee, each trade weighted equally
    (paper sizes are uncapped, so dollar PnL is not comparable to live)."""
    settled = [t for t in trades if t.realized_pnl_cents is not None and t.filled_count > 0 and t.side]
    if not settled:
        return None
    won = np.array([1.0 if t.settled_yes == (t.side == "yes") else 0.0 for t in settled])
    price = np.array([t.avg_fill_price_cents or 0.0 for t in settled])
    fee = np.array([t.fees_cents / t.filled_count for t in settled])
    edge = won * 100 - price - fee
    half_width = 1.96 * float(edge.std(ddof=1)) / math.sqrt(len(edge)) if len(edge) > 1 else float("inf")
    return EdgeReport(
        mode=mode,
        trades=len(settled),
        win_rate=float(won.mean()),
        avg_price_cents=float(price.mean()),
        avg_fee_cents=float(fee.mean()),
        edge_cents=float(edge.mean()),
        ci_low_cents=float(edge.mean()) - half_width,
        ci_high_cents=float(edge.mean()) + half_width,
        total_pnl=sum(t.realized_pnl_cents or 0 for t in settled) / 100,
    )


def llm_value(rows: Sequence[tuple[EvaluationTraceRecord, bool]]) -> Optional[LLMValue]:
    """`rows` pairs each LLM-reviewed decision with whether its contract settled YES."""
    reviewed = [(e, yes) for e, yes in rows if e.llm_should_bet is not None and e.proposed_side and e.proposed_ask_cents]
    if not reviewed:
        return None

    approved = [_hypothetical_cents(e, y) for e, y in reviewed if e.llm_should_bet]
    vetoed = [_hypothetical_cents(e, y) for e, y in reviewed if not e.llm_should_bet]
    y = np.array([1.0 if yes else 0.0 for _, yes in reviewed])
    p_llm = np.array([e.llm_adjusted_probability if e.llm_adjusted_probability is not None else np.nan for e, _ in reviewed])
    p_model = np.array([e.model_probability for e, _ in reviewed])
    has_llm = ~np.isnan(p_llm)
    return LLMValue(
        approved=len(approved),
        vetoed=len(vetoed),
        approved_edge_cents=float(np.mean(approved)) if approved else None,
        vetoed_edge_cents=float(np.mean(vetoed)) if vetoed else None,
        saved_cents=-float(np.sum(vetoed)) if vetoed else 0.0,
        brier_llm=brier(p_llm[has_llm], y[has_llm]) if has_llm.any() else None,
        brier_model=brier(p_model, y),
    )


def _summary(result: BacktestResult) -> StrategySummary:
    return StrategySummary(
        params=result.params, trades=result.num_trades, total_net_pnl=result.total_net_pnl,
        sharpe_ratio=result.sharpe_ratio, win_rate=result.win_rate,
    )


def holdout_backtest(
    ticks: Sequence[Any],
    trades: Sequence[TradeRecord],
    current: StrategyParameters,
    *,
    grid: Optional[ParameterGrid] = None,
    min_trades: int = 300,
    test_fraction: float = 0.3,
) -> tuple[HoldoutBacktest, Optional[StrategyParameters]]:
    """Tune on the older 70% of replayable trades, compare against the active parameters on the
    newest 30%. Returns the report and the parameters to apply (if they clearly win)."""
    ordered = sorted(trades, key=lambda t: t.entry_time)
    replayable = Backtester(ticks, ordered).num_replayable_trades
    if replayable < min_trades:
        return HoldoutBacktest(status=f"waiting for data: {replayable} of {min_trades} replayable trades"), None
    split = int(len(ordered) * (1 - test_fraction))
    train, test = ordered[:split], ordered[split:]
    # search() only replays in memory; the engine is used by run()/save(), which we don't call.
    tuner = Optimizer(None, grid=grid, base_params=current, min_trades=max(10, min_trades // 10))  # type: ignore[arg-type]
    ranked = tuner.search(Backtester(ticks, train))
    test_bt = Backtester(ticks, test)
    current_result = test_bt.run(current)
    report = HoldoutBacktest(
        status="ok", train_trades=len(train), test_trades=len(test), current=_summary(current_result)
    )
    if not ranked:
        report.status = "no parameter set traded enough on the training split"
        return report, None
    best_result = test_bt.run(ranked[0].params)
    report.best = _summary(best_result)
    clearly_better = (
        best_result.num_trades >= 30
        and best_result.total_net_pnl > current_result.total_net_pnl
        and best_result.sharpe_ratio > current_result.sharpe_ratio + 0.5
        and ranked[0].params != current
    )
    return report, (ranked[0].params if clearly_better else None)


# ----------------------------------------------------------------------------- job


async def build_report(db: Database, now: datetime, *, apply_params: bool, cooldown_hours: float, min_trades: int) -> EvaluationReport:
    since = now - timedelta(days=LOOKBACK_DAYS)
    notes: list[str] = []
    async with db.session() as session:
        settled_snapshots = (
            select(func.count())
            .select_from(MarketSnapshot)
            .join(ContractOutcome, ContractOutcome.ticker == MarketSnapshot.ticker)
            .where(MarketSnapshot.ts >= since)
        )
        stride = max(1, math.ceil((await session.exec(settled_snapshots)).one() / MAX_SNAPSHOT_ROWS))
        snap_rows = (
            await session.exec(
                select(
                    MarketSnapshot.ticker, MarketSnapshot.seconds_remaining, MarketSnapshot.yes_bid,
                    MarketSnapshot.yes_ask, MarketSnapshot.model_probability, ContractOutcome.settled_yes,
                    MarketSnapshot.spot_model_probability, MarketSnapshot.settlement_model_probability,
                    MarketSnapshot.momentum_1m, MarketSnapshot.historical_volatility,
                    MarketSnapshot.order_book_imbalance, MarketSnapshot.close_time,
                )
                .join(ContractOutcome, ContractOutcome.ticker == MarketSnapshot.ticker)
                .where(MarketSnapshot.ts >= since, col(MarketSnapshot.id) % stride == 0)
                .order_by(col(MarketSnapshot.close_time), col(MarketSnapshot.ts))
            )
        ).all()
        maker_rows = (
            await session.exec(
                select(MakerShadowOrder, ContractOutcome.settled_yes)
                .join(ContractOutcome, ContractOutcome.ticker == MakerShadowOrder.ticker)
                .where(MakerShadowOrder.created_at >= since, col(MakerShadowOrder.filled).is_not(None))
            )
        ).all()
        replay_rows = (
            await session.exec(
                select(EvaluationTraceRecord, LLMReplayResult, ContractOutcome.settled_yes)
                .join(LLMReplayResult, LLMReplayResult.trace_id == EvaluationTraceRecord.id)
                .join(ContractOutcome, ContractOutcome.ticker == EvaluationTraceRecord.ticker)
                .where(EvaluationTraceRecord.created_at >= since)
            )
        ).all()
        trades = list((await session.exec(select(TradeRecord).where(TradeRecord.entry_time >= since))).all())
        outcomes = {o.ticker: o.settled_yes for o in (await session.exec(select(ContractOutcome))).all()}
        traces = (
            await session.exec(
                select(EvaluationTraceRecord).where(
                    EvaluationTraceRecord.created_at >= since, col(EvaluationTraceRecord.llm_should_bet).is_not(None)
                )
            )
        ).all()
        ticks = [r.to_tick() for r in (await session.exec(
            select(BtcPriceHistory).where(BtcPriceHistory.timestamp >= since).order_by(col(BtcPriceHistory.timestamp))
        )).all()]
        last_applied = (
            await session.exec(
                select(StrategyParametersRecord.created_at)
                .where(StrategyParametersRecord.source == "holdout")
                .order_by(col(StrategyParametersRecord.created_at).desc())
            )
        ).first()
    # Trade outcomes also count (contracts settled before snapshots were recorded).
    for t in trades:
        if t.settled_yes is not None:
            outcomes.setdefault(t.ticker, t.settled_yes)

    scores: list[ProbabilityScore] = []
    best_w: Optional[float] = None
    best_model: Optional[str] = None
    blend_weights: dict[str, float] = {}
    model_contracts = 0
    buckets: list[CalibrationBucket] = []
    horizons: list[HorizonScore] = []
    learned: Optional[LearnedModelReport] = None
    contracts = len({r[0] for r in snap_rows})
    if snap_rows:
        # Compare the two models on the same rows: those recorded after the settlement model existed.
        # Older rows are spot-only (their model_probability was the spot model).
        both = [r for r in snap_rows if r[7] is not None]
        rows = both or list(snap_rows)
        model_contracts = len({r[0] for r in rows})
        seconds = np.array([r[1] for r in rows], dtype=float)
        p_market = _clip(np.array([(r[2] + r[3]) / 200 for r in rows], dtype=float))
        p_spot = _clip(np.array([r[6] if r[6] is not None else r[4] for r in rows], dtype=float))
        y = np.array([1.0 if r[5] else 0.0 for r in rows])
        models: dict[str, np.ndarray] = {"spot": p_spot}
        if both:
            models["settlement"] = _clip(np.array([r[7] for r in rows], dtype=float))
        scores = [score("Market (mid price)", p_market, y)]
        scores += [score(MODEL_LABELS[name], p, y) for name, p in models.items()]
        best_model = min(models, key=lambda name: log_loss(models[name], y))
        blend_weights = {name: fit_blend_weight(p, p_market, y) for name, p in models.items()}
        best_w = blend_weights[best_model]
        p_best = models[best_model]
        scores.append(score(f"Blend ({best_w:.0%} {best_model} model + market)", best_w * p_best + (1 - best_w) * p_market, y))
        buckets = calibration(p_best, y)
        horizons = horizon_scores(seconds, p_spot, p_market, y, models.get("settlement"))

        close_rank = {t: i for i, t in enumerate(dict.fromkeys(r[0] for r in rows))}  # rows are sorted by close
        x = learned_features(
            p_market, p_best, seconds,
            np.array([r[8] for r in rows], dtype=float), np.array([r[9] for r in rows], dtype=float),
            np.array([r[10] for r in rows], dtype=float),
        )
        order = np.array([close_rank[r[0]] for r in rows])
        learned = await asyncio.to_thread(walk_forward_learned, order, x, y, p_market, p_best)
        notes.append(
            f"{len(rows):,} snapshots from {model_contracts} settled contracts. Snapshots within one contract are "
            "highly correlated, so the effective sample size is closer to the contract count."
        )
        if stride > 1:
            notes.append(f"Sampled every {stride}th snapshot to bound memory.")
        if not both:
            notes.append("No snapshots with the settlement model yet: scores cover the spot model only.")
    else:
        notes.append("No settled contracts with market snapshots yet (snapshots started with this release).")

    edges = [e for e in (realized_edge([t for t in trades if t.is_paper], "paper"),
                         realized_edge([t for t in trades if not t.is_paper], "live")) if e]
    llm = llm_value([(e, outcomes[e.ticker]) for e in traces if e.ticker in outcomes])

    current = await load_active_parameters(db.engine)
    backtest, to_apply = await asyncio.to_thread(holdout_backtest, ticks, trades, current, min_trades=min_trades)
    if to_apply is not None:
        recent = last_applied is not None and now - (
            last_applied if last_applied.tzinfo else last_applied.replace(tzinfo=timezone.utc)
        ) < timedelta(hours=cooldown_hours)
        if not apply_params:
            backtest.status = "better parameters found (auto-apply disabled)"
        elif recent:
            backtest.status = f"better parameters found; last change < {cooldown_hours:.0f}h ago, waiting"
        else:
            async with db.session() as session:
                for row in (await session.exec(select(StrategyParametersRecord).where(StrategyParametersRecord.is_active))).all():
                    row.is_active = False
                    session.add(row)
                session.add(StrategyParametersRecord.from_schema(
                    to_apply, is_active=True, source="holdout", notes="beat active params on held-out trades",
                    sharpe_ratio=backtest.best.sharpe_ratio if backtest.best else None,
                    total_net_pnl=backtest.best.total_net_pnl if backtest.best else None,
                    num_trades=backtest.best.trades if backtest.best else None,
                ))
                await session.commit()
            backtest.applied = True
            backtest.status = "applied new parameters (beat active ones on held-out trades)"
    return EvaluationReport(
        created_at=now, contracts=contracts, snapshots=len(snap_rows), scores=scores, best_blend_weight=best_w,
        best_model=best_model, model_contracts=model_contracts, blend_weights=blend_weights,
        calibration=buckets, horizons=horizons, edges=edges, llm=llm, backtest=backtest, notes=notes,
        learned=learned, maker=maker_reports(list(maker_rows)), llm_replays=llm_replay_summaries(list(replay_rows)),
    )


async def run_once(
    database_url: str,
    *,
    apply_params: bool = True,
    cooldown_hours: float = 24.0,
    min_trades: int = 300,
    replay_model: Optional[str] = None,
    replay_effort: str = "low",
    replay_daily_usd: float = 0.0,
) -> EvaluationReport:
    db = Database(database_url)
    try:
        await db.init()
        if replay_model and replay_daily_usd > 0:
            try:
                await run_replays(
                    db, model=replay_model, effort=replay_effort, daily_budget_usd=replay_daily_usd,
                    now=datetime.now(timezone.utc),
                )
            except Exception:
                logger.exception("LLM replays failed; continuing with the report")
        report = await build_report(
            db, datetime.now(timezone.utc), apply_params=apply_params, cooldown_hours=cooldown_hours, min_trades=min_trades
        )
        async with db.session() as session:
            session.add(EvaluationReportRecord(created_at=report.created_at, report_json=report.model_dump_json()))
            await session.commit()
        return report
    finally:
        await db.dispose()


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s evaluation: %(message)s")
    with_nice = getattr(os, "nice", None)
    if with_nice is not None:
        with_nice(15)  # yield the CPU to the live trading process
    report = asyncio.run(
        run_once(
            os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
            apply_params=os.environ.get("AUTO_APPLY_PARAMS", "true").lower() in ("1", "true", "yes"),
            cooldown_hours=float(os.environ.get("AUTO_APPLY_COOLDOWN_HOURS", "24")),
            min_trades=int(os.environ.get("AUTO_APPLY_MIN_TRADES", "300")),
            replay_model=os.environ.get("LLM_REPLAY_MODEL") if os.environ.get("ANTHROPIC_API_KEY") else None,
            replay_effort=os.environ.get("LLM_REPLAY_EFFORT", "low"),
            replay_daily_usd=float(os.environ.get("LLM_REPLAY_DAILY_USD", "0")),
        )
    )
    logger.info("report: %d contracts, %d snapshots; backtest: %s", report.contracts, report.snapshots, report.backtest.status)


if __name__ == "__main__":
    main()
