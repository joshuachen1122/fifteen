"""Live performance and LLM-spend summary for the dashboard, computed from SQLite."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from db.models import EvaluationTraceRecord, TradeRecord

HOURS_PER_MONTH = 30 * 24


class AgentStats(BaseModel):
    updated_at: datetime
    mode: str  # "paper" | "live"
    halted: bool

    # Bankroll & PnL (USD; PnL is realized and net of Kalshi fees)
    equity: float
    total_pnl: float
    return_pct: Optional[float]
    fees_paid: float
    open_positions: int
    open_exposure: float

    # Bets
    bets_made: int
    bets_settled: int
    wins: int
    win_rate_pct: Optional[float]
    total_staked: float
    roi_on_staked_pct: Optional[float]

    # LLM gate
    candidates_screened: int  # rejected by free checks, never sent to the LLM
    screened_by_reason: dict[str, int]
    llm_calls: int
    llm_calls_today: int
    llm_approvals: int
    approval_rate_pct: Optional[float]
    bet_per_call_pct: Optional[float]  # share of LLM calls that ended in a filled bet

    # LLM spend (USD)
    llm_cost_total: float
    llm_cost_today: float
    llm_daily_budget: float
    budget_used_pct: float
    avg_cost_per_call: Optional[float]
    projected_monthly_cost: Optional[float]
    projection_basis_hours: float
    cache_hit_pct: Optional[float]
    pnl_after_llm: float
    llm_cost_pct_of_pnl: Optional[float]


def _pct(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator * 100 if denominator else None


async def compute_stats(
    session: AsyncSession,
    *,
    now: datetime,
    is_paper: bool,
    initial_bankroll_cents: int,
    equity_cents: int,
    open_positions: int,
    open_exposure_cents: int,
    halted: bool,
    llm_daily_budget_usd: float,
    llm_screened: dict[str, int],
) -> AgentStats:
    bets = (
        await session.exec(
            select(TradeRecord).where(TradeRecord.is_paper == is_paper, TradeRecord.filled_count > 0)
        )
    ).all()
    settled = [b for b in bets if b.realized_pnl_cents is not None]
    pnl_cents = sum(b.realized_pnl_cents or 0 for b in settled)
    staked_cents = sum(round((b.avg_fill_price_cents or 0) * b.filled_count) for b in settled)
    wins = sum(1 for b in settled if (b.realized_pnl_cents or 0) > 0)

    called = col(EvaluationTraceRecord.llm_cost_usd).is_not(None)
    calls, approvals, cost, uncached, cache_read, cache_write, first_call = (
        await session.exec(
            select(
                func.count(),
                func.coalesce(func.sum(EvaluationTraceRecord.llm_should_bet), 0),
                func.coalesce(func.sum(EvaluationTraceRecord.llm_cost_usd), 0.0),
                func.coalesce(func.sum(EvaluationTraceRecord.llm_input_tokens), 0),
                func.coalesce(func.sum(EvaluationTraceRecord.llm_cache_read_tokens), 0),
                func.coalesce(func.sum(EvaluationTraceRecord.llm_cache_write_tokens), 0),
                func.min(EvaluationTraceRecord.created_at),
            ).where(called)
        )
    ).one()

    midnight = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
    calls_today, cost_today = (
        await session.exec(
            select(func.count(), func.coalesce(func.sum(EvaluationTraceRecord.llm_cost_usd), 0.0)).where(
                called, EvaluationTraceRecord.created_at >= midnight
            )
        )
    ).one()

    # Project monthly spend from the trailing 24h, or from however long we have been calling the LLM.
    projected: Optional[float] = None
    basis_hours = 0.0
    if first_call is not None:
        first = first_call if first_call.tzinfo else first_call.replace(tzinfo=timezone.utc)
        window_start = max(first, now - timedelta(hours=24))
        basis_hours = max((now - window_start).total_seconds() / 3600, 1 / 60)
        window_cost = (
            await session.exec(
                select(func.coalesce(func.sum(EvaluationTraceRecord.llm_cost_usd), 0.0)).where(
                    called, EvaluationTraceRecord.created_at >= window_start
                )
            )
        ).one()
        projected = float(window_cost) / basis_hours * HOURS_PER_MONTH

    total_pnl = pnl_cents / 100
    llm_cost = float(cost)
    return AgentStats(
        updated_at=now,
        mode="paper" if is_paper else "live",
        halted=halted,
        equity=equity_cents / 100,
        total_pnl=total_pnl,
        return_pct=_pct(pnl_cents, initial_bankroll_cents),
        fees_paid=sum(b.fees_cents for b in bets) / 100,
        open_positions=open_positions,
        open_exposure=open_exposure_cents / 100,
        bets_made=len(bets),
        bets_settled=len(settled),
        wins=wins,
        win_rate_pct=_pct(wins, len(settled)),
        total_staked=staked_cents / 100,
        roi_on_staked_pct=_pct(pnl_cents, staked_cents),
        candidates_screened=sum(llm_screened.values()),
        screened_by_reason=dict(sorted(llm_screened.items(), key=lambda kv: -kv[1])),
        llm_calls=calls,
        llm_calls_today=calls_today,
        llm_approvals=int(approvals),
        approval_rate_pct=_pct(int(approvals), calls),
        bet_per_call_pct=_pct(len(bets), calls),
        llm_cost_total=llm_cost,
        llm_cost_today=float(cost_today),
        llm_daily_budget=llm_daily_budget_usd,
        budget_used_pct=float(cost_today) / llm_daily_budget_usd * 100 if llm_daily_budget_usd else 0.0,
        avg_cost_per_call=llm_cost / calls if calls else None,
        projected_monthly_cost=projected,
        projection_basis_hours=basis_hours,
        cache_hit_pct=_pct(cache_read, uncached + cache_read + cache_write),
        pnl_after_llm=total_pnl - llm_cost,
        llm_cost_pct_of_pnl=_pct(llm_cost, total_pnl) if total_pnl > 0 else None,
    )
