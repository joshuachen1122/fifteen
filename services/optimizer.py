"""Learning sub-agent: grid-searches StrategyParameters and publishes the winner.

The live trader picks up new parameters via `load_active_parameters`.
"""

import asyncio
import itertools
import logging
from typing import Optional, Sequence

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col, select, update
from sqlmodel.ext.asyncio.session import AsyncSession

from core.schemas import BtcSpotTick, StrategyParameters
from db.models import BtcPriceHistory, StrategyParametersRecord, TradeRecord
from services.backtester import Backtester, BacktestResult

logger = logging.getLogger(__name__)


def _frange(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 6) for i in range(count)]


class ParameterGrid(BaseModel):
    min_gross_edge: list[float] = Field(default_factory=lambda: _frange(0.02, 0.08, 0.005))
    volatility_lookback_sec: list[int] = Field(default_factory=lambda: list(range(60, 901, 60)))
    momentum_weight: list[float] = Field(default_factory=lambda: _frange(0.0, 0.4, 0.05))

    @property
    def size(self) -> int:
        return len(self.min_gross_edge) * len(self.volatility_lookback_sec) * len(self.momentum_weight)

    def candidates(self, base: StrategyParameters) -> list[StrategyParameters]:
        return [
            base.model_copy(update={"min_gross_edge": e, "volatility_lookback_sec": v, "momentum_weight": m})
            for e, v, m in itertools.product(self.min_gross_edge, self.volatility_lookback_sec, self.momentum_weight)
        ]


class OptimizationResult(BaseModel):
    evaluated: int
    replayable_trades: int
    best: Optional[BacktestResult]
    leaderboard: list[BacktestResult]
    record_id: Optional[int] = None
    message: str


def _rank_key(result: BacktestResult) -> tuple[float, float, float]:
    return (result.sharpe_ratio, result.total_net_pnl, result.win_rate)


class Optimizer:
    def __init__(
        self,
        engine: AsyncEngine,
        grid: Optional[ParameterGrid] = None,
        base_params: Optional[StrategyParameters] = None,
        min_trades: int = 30,
        leaderboard_size: int = 10,
        starting_bankroll_cents: int = 100_000,
    ) -> None:
        self.engine = engine
        self.grid = grid or ParameterGrid()
        self.base_params = base_params or StrategyParameters()
        self.min_trades = min_trades
        self.leaderboard_size = leaderboard_size
        self.starting_bankroll_cents = starting_bankroll_cents

    def search(self, backtester: Backtester) -> list[BacktestResult]:
        """Backtest every grid point; return qualifying results, best first."""
        results = [backtester.run(p) for p in self.grid.candidates(self.base_params)]
        qualifying = [r for r in results if r.num_trades >= self.min_trades]
        qualifying.sort(key=_rank_key, reverse=True)
        return qualifying

    async def run(
        self,
        ticks: Optional[Sequence[BtcSpotTick]] = None,
        trades: Optional[Sequence[TradeRecord]] = None,
        save: bool = True,
    ) -> OptimizationResult:
        if ticks is None or trades is None:
            db_ticks, db_trades = await self._load_history()
            ticks = db_ticks if ticks is None else ticks
            trades = db_trades if trades is None else trades

        backtester = Backtester(ticks, trades, self.starting_bankroll_cents)
        ranked = await asyncio.to_thread(self.search, backtester)
        leaderboard = ranked[: self.leaderboard_size]
        if not leaderboard:
            message = (
                f"No parameter set produced at least {self.min_trades} trades "
                f"({backtester.num_replayable_trades} replayable trades in history)."
            )
            logger.warning(message)
            return OptimizationResult(
                evaluated=self.grid.size,
                replayable_trades=backtester.num_replayable_trades,
                best=None,
                leaderboard=[],
                message=message,
            )

        best = leaderboard[0]
        record_id = (await self.save(best)).id if save else None
        return OptimizationResult(
            evaluated=self.grid.size,
            replayable_trades=backtester.num_replayable_trades,
            best=best,
            leaderboard=leaderboard,
            record_id=record_id,
            message=f"Best of {self.grid.size} parameter sets: Sharpe {best.sharpe_ratio:.2f}.",
        )

    async def save(self, result: BacktestResult) -> StrategyParametersRecord:
        """Insert the winning parameters as the single active row."""
        p = result.params
        record = StrategyParametersRecord(
            min_gross_edge=p.min_gross_edge,
            volatility_lookback_sec=p.volatility_lookback_sec,
            momentum_weight=p.momentum_weight,
            kelly_scaling=p.kelly_scaling,
            max_trade_bankroll_pct=p.max_trade_bankroll_pct,
            sharpe_ratio=result.sharpe_ratio,
            win_rate=result.win_rate,
            total_net_pnl=result.total_net_pnl,
            num_trades=result.num_trades,
            is_active=True,
            source="optimizer",
        )
        async with AsyncSession(self.engine, expire_on_commit=False) as session:
            await session.exec(update(StrategyParametersRecord).values(is_active=False))
            session.add(record)
            await session.commit()
            await session.refresh(record)
        return record

    async def _load_history(self) -> tuple[list[BtcSpotTick], list[TradeRecord]]:
        async with AsyncSession(self.engine) as session:
            tick_rows = (await session.exec(select(BtcPriceHistory).order_by(BtcPriceHistory.timestamp))).all()
            trade_rows = (await session.exec(select(TradeRecord).order_by(TradeRecord.entry_time))).all()
        return [r.to_tick() for r in tick_rows], list(trade_rows)


async def load_active_parameters(engine: AsyncEngine) -> StrategyParameters:
    """Latest optimizer-published parameters, or the schema defaults if none exist."""
    async with AsyncSession(engine) as session:
        statement = (
            select(StrategyParametersRecord)
            .where(col(StrategyParametersRecord.is_active))
            .order_by(col(StrategyParametersRecord.created_at).desc(), col(StrategyParametersRecord.id).desc())
        )
        record = (await session.exec(statement)).first()
    return record.to_schema() if record else StrategyParameters()
