"""Hourly, report-only evaluation of the model, the LLM gate, and realized trading edge.

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
from sqlmodel import col, select

from core.constants import DEFAULT_DATABASE_URL
from core.schemas import StrategyParameters
from db.database import Database
from db.models import (
    BtcPriceHistory,
    ContractOutcome,
    EvaluationReportRecord,
    EvaluationTraceRecord,
    MarketSnapshot,
    StrategyParametersRecord,
    TradeRecord,
)
from services.backtester import Backtester, BacktestResult
from services.optimizer import Optimizer, ParameterGrid, load_active_parameters
from services.risk_engine import kalshi_taker_fee_cents

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 30
PROB_CLIP = 0.005  # keeps log loss finite when the market quotes 0c/100c
HORIZONS = [(0, 60, "< 1 min"), (60, 180, "1-3 min"), (180, 420, "3-7 min"), (420, 10_000, "7-15 min")]


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
    brier_model: float


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
    best_blend_weight: Optional[float]  # weight on the model in w*model + (1-w)*market
    calibration: list[CalibrationBucket]
    horizons: list[HorizonScore]
    edges: list[EdgeReport]
    llm: Optional[LLMValue]
    backtest: HoldoutBacktest
    notes: list[str]


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
    seconds: np.ndarray, p_model: np.ndarray, p_market: np.ndarray, y: np.ndarray
) -> list[HorizonScore]:
    out = []
    for lo, hi, label in HORIZONS:
        mask = (seconds >= lo) & (seconds < hi)
        if mask.any():
            out.append(
                HorizonScore(
                    label=label, n=int(mask.sum()),
                    brier_market=brier(p_market[mask], y[mask]), brier_model=brier(p_model[mask], y[mask]),
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

    def hypothetical(e: EvaluationTraceRecord, settled_yes: bool) -> float:
        ask = e.proposed_ask_cents or 0
        won = settled_yes == (e.proposed_side == "yes")
        return (100.0 if won else 0.0) - ask - kalshi_taker_fee_cents(1, ask)

    approved = [hypothetical(e, y) for e, y in reviewed if e.llm_should_bet]
    vetoed = [hypothetical(e, y) for e, y in reviewed if not e.llm_should_bet]
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
        snap_rows = (
            await session.exec(
                select(
                    MarketSnapshot.ticker, MarketSnapshot.seconds_remaining, MarketSnapshot.yes_bid,
                    MarketSnapshot.yes_ask, MarketSnapshot.model_probability, ContractOutcome.settled_yes,
                )
                .join(ContractOutcome, ContractOutcome.ticker == MarketSnapshot.ticker)
                .where(MarketSnapshot.ts >= since)
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
    buckets: list[CalibrationBucket] = []
    horizons: list[HorizonScore] = []
    contracts = len({r[0] for r in snap_rows})
    if snap_rows:
        seconds = np.array([r[1] for r in snap_rows], dtype=float)
        p_market = _clip(np.array([(r[2] + r[3]) / 200 for r in snap_rows], dtype=float))
        p_model = _clip(np.array([r[4] for r in snap_rows], dtype=float))
        y = np.array([1.0 if r[5] else 0.0 for r in snap_rows])
        best_w = fit_blend_weight(p_model, p_market, y)
        scores = [
            score("Market (mid price)", p_market, y),
            score("Our model", p_model, y),
            score(f"Blend (w={best_w:.2f} on model)", best_w * p_model + (1 - best_w) * p_market, y),
        ]
        buckets = calibration(p_model, y)
        horizons = horizon_scores(seconds, p_model, p_market, y)
        notes.append(
            f"{len(snap_rows):,} snapshots from {contracts} settled contracts. Snapshots within one contract are "
            "highly correlated, so the effective sample size is closer to the contract count."
        )
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
        calibration=buckets, horizons=horizons, edges=edges, llm=llm, backtest=backtest, notes=notes,
    )


async def run_once(database_url: str, *, apply_params: bool = True, cooldown_hours: float = 24.0, min_trades: int = 300) -> EvaluationReport:
    db = Database(database_url)
    try:
        await db.init()
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
        )
    )
    logger.info("report: %d contracts, %d snapshots; backtest: %s", report.contracts, report.snapshots, report.backtest.status)


if __name__ == "__main__":
    main()
