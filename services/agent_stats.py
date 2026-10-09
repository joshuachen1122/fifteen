"""Live performance and LLM-spend summary for the dashboard, computed from SQLite."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import Integer, cast, func
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
    bet_per_call_pct: Optional[float]  # share of tracked LLM calls that ended in a filled bet

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
    # Cast before summing: SQLAlchemy types SUM(boolean) as Boolean, which collapses any count to True.
    calls, approvals, traded, cost, uncached, cache_read, cache_write, first_call = (
        await session.exec(
            select(
                func.count(),
                func.coalesce(func.sum(cast(EvaluationTraceRecord.llm_should_bet, Integer)), 0),
                func.coalesce(func.sum(cast(EvaluationTraceRecord.decision == "trade", Integer)), 0),
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
        bet_per_call_pct=_pct(int(traded), calls),
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


class OpenPosition(BaseModel):
    ticker: str
    side: str
    count: int
    price_cents: float
    cost: float  # USD including fees
    entered_at: datetime
    expires_at: Optional[datetime]
    seconds_left: Optional[int]
    model_win_probability: Optional[float]
    mark_cents: Optional[int]  # best bid on our side: what we could sell for now
    unrealized_pnl: Optional[float]  # USD, after the entry fee, if sold at the mark


class OrderFill(BaseModel):
    at: datetime
    ticker: str
    side: str
    requested: int
    filled: int
    price_cents: Optional[float]
    fees: float
    status: str
    order_id: Optional[str]
    outcome: Optional[str]  # "won" | "lost" | None while open
    pnl: Optional[float]


class Decision(BaseModel):
    at: datetime
    ticker: str
    side: Optional[str]
    ask_cents: Optional[int]
    seconds_remaining: int
    edge: float
    model_probability: float  # P(YES)
    llm_probability: Optional[float]  # P(YES)
    confidence: Optional[float]
    approved: Optional[bool]
    decision: str  # "trade" | "skip"
    skip_reason: Optional[str]
    reasoning: Optional[str]
    risk_signals: list[str]
    cost_usd: Optional[float]


class AgentActivity(BaseModel):
    updated_at: datetime
    positions: list[OpenPosition]
    orders: list[OrderFill]
    decisions: list[Decision]


def _aware(moment: Optional[datetime]) -> Optional[datetime]:
    return moment if moment is None or moment.tzinfo else moment.replace(tzinfo=timezone.utc)


async def compute_activity(
    session: AsyncSession,
    *,
    now: datetime,
    is_paper: bool,
    books: dict[str, tuple[Optional[int], Optional[int]]],  # ticker -> (best yes bid, best yes ask)
    limit: int = 20,
) -> AgentActivity:
    open_rows = (
        await session.exec(
            select(TradeRecord)
            .where(
                TradeRecord.is_paper == is_paper,
                TradeRecord.filled_count > 0,
                col(TradeRecord.settled_yes).is_(None),
            )
            .order_by(col(TradeRecord.entry_time).desc())
        )
    ).all()
    positions = []
    for r in open_rows:
        yes_bid, yes_ask = books.get(r.ticker, (None, None))
        mark = yes_bid if r.side == "yes" else (None if yes_ask is None else 100 - yes_ask)
        price = r.avg_fill_price_cents or 0.0
        expires = _aware(r.expiration_time)
        p_yes = r.model_probability
        positions.append(
            OpenPosition(
                ticker=r.ticker,
                side=r.side or "",
                count=r.filled_count,
                price_cents=price,
                cost=(price * r.filled_count + r.fees_cents) / 100,
                entered_at=_aware(r.entry_time) or now,
                expires_at=expires,
                seconds_left=max(int((expires - now).total_seconds()), 0) if expires else None,
                model_win_probability=None if p_yes is None else (p_yes if r.side == "yes" else 1 - p_yes),
                mark_cents=mark,
                unrealized_pnl=None if mark is None else ((mark - price) * r.filled_count - r.fees_cents) / 100,
            )
        )

    order_rows = (
        await session.exec(
            select(TradeRecord)
            .where(TradeRecord.is_paper == is_paper, col(TradeRecord.side).is_not(None))
            .order_by(col(TradeRecord.entry_time).desc())
            .limit(limit)
        )
    ).all()
    orders = [
        OrderFill(
            at=_aware(r.entry_time) or now,
            ticker=r.ticker,
            side=r.side or "",
            requested=r.count,
            filled=r.filled_count,
            price_cents=r.avg_fill_price_cents,
            fees=r.fees_cents / 100,
            status=r.status,
            order_id=r.kalshi_order_id,
            outcome=None if r.realized_pnl_cents is None else ("won" if r.settled_yes == (r.side == "yes") else "lost"),
            pnl=None if r.realized_pnl_cents is None else r.realized_pnl_cents / 100,
        )
        for r in order_rows
    ]

    trace_rows = (
        await session.exec(
            select(EvaluationTraceRecord).order_by(col(EvaluationTraceRecord.created_at).desc()).limit(limit)
        )
    ).all()
    decisions = [
        Decision(
            at=_aware(r.created_at) or now,
            ticker=r.ticker,
            side=r.proposed_side,
            ask_cents=r.proposed_ask_cents,
            seconds_remaining=r.seconds_remaining,
            edge=r.gross_edge,
            model_probability=r.model_probability,
            llm_probability=r.llm_adjusted_probability,
            confidence=r.llm_confidence_score,
            approved=r.llm_should_bet,
            decision=r.decision,
            skip_reason=r.skip_reason,
            reasoning=r.llm_reasoning,
            risk_signals=list(r.llm_risk_signals or []),
            cost_usd=r.llm_cost_usd,
        )
        for r in trace_rows
    ]
    return AgentActivity(updated_at=now, positions=positions, orders=orders, decisions=decisions)
