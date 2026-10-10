"""LLM cost experiment: re-ask logged risk-gate decisions through a cheaper model.

Each live LLM decision stores its exact input (`EvaluationTraceRecord.llm_input_json`). Here the
same input is sent to another model (default Claude Haiku 5.5 at low effort) and the verdict is
stored next to the original, so the hourly report can compare agreement, accuracy against the
settled outcome, and cost per call. Only settled contracts are replayed (they can be scored), and
spend is capped per UTC day. Runs inside the hourly evaluation process.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, time, timezone
from typing import Optional

from sqlalchemy import func
from sqlmodel import col, select

from core.schemas import FairValueOutput
from db.database import Database
from db.models import ContractOutcome, EvaluationTraceRecord, LLMReplayResult
from services.llm_evaluator import LLMEvaluator

logger = logging.getLogger(__name__)

FAIL_CLOSED_SIGNALS = {"LLM unavailable", "LLM invalid output", "LLM refusal"}


async def replay_spend_today(db: Database, model: str, now: datetime) -> float:
    midnight = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
    async with db.session() as session:
        spent = (
            await session.exec(
                select(func.coalesce(func.sum(LLMReplayResult.cost_usd), 0.0)).where(
                    LLMReplayResult.model == model, LLMReplayResult.created_at >= midnight
                )
            )
        ).one()
    return float(spent)


async def run_replays(
    db: Database,
    *,
    model: str,
    effort: str,
    daily_budget_usd: float,
    now: datetime,
    max_calls: int = 60,
    evaluator_factory: Optional[Callable[[str, str], LLMEvaluator]] = None,
) -> int:
    """Replay settled, not-yet-replayed decisions (newest first) until the daily budget is spent."""
    if daily_budget_usd <= 0:
        return 0
    spent = await replay_spend_today(db, model, now)
    if spent >= daily_budget_usd:
        return 0
    async with db.session() as session:
        replayed = select(LLMReplayResult.trace_id).where(LLMReplayResult.model == model)
        traces = list(
            (
                await session.exec(
                    select(EvaluationTraceRecord)
                    .join(ContractOutcome, ContractOutcome.ticker == EvaluationTraceRecord.ticker)
                    .where(
                        col(EvaluationTraceRecord.llm_input_json).is_not(None),
                        col(EvaluationTraceRecord.llm_should_bet).is_not(None),
                        col(EvaluationTraceRecord.id).not_in(replayed),
                    )
                    .order_by(col(EvaluationTraceRecord.created_at).desc())
                    .limit(max_calls)
                )
            ).all()
        )
    if not traces:
        return 0

    evaluator = (evaluator_factory or (lambda m, e: LLMEvaluator(model=m, effort=e)))(model, effort)
    done = 0
    for trace in traces:
        if spent >= daily_budget_usd:
            break
        payload = json.loads(trace.llm_input_json or "{}")
        candidate = FairValueOutput.model_validate(payload["candidate"])
        verdict, usage = await evaluator.evaluate_with_usage(candidate, payload["context"])
        cost = usage.cost_usd if usage else None
        spent += cost or 0.0
        assert trace.id is not None
        async with db.session() as session:
            session.add(
                LLMReplayResult(
                    created_at=now,
                    trace_id=trace.id,
                    model=model,
                    effort=effort,
                    should_bet=verdict.should_bet,
                    adjusted_probability=verdict.adjusted_probability,
                    failed=usage is None or bool(FAIL_CLOSED_SIGNALS & set(verdict.risk_signals)),
                    cost_usd=cost,
                )
            )
            await session.commit()
        done += 1
    logger.info("replayed %d decision(s) through %s (%s effort), $%.4f spent today", done, model, effort, spent)
    return done
