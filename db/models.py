"""SQLite persistence models shared by the backtester, optimizer, and dashboard.

These are storage rows, not part of the `core/schemas.py` contract. Conversion
helpers map them onto the contract types where one exists.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel

from core.schemas import BtcSpotTick, StrategyParameters


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BtcPriceTickRecord(SQLModel, table=True):
    """Historical 1-minute BTC spot ticks used to replay volatility & momentum."""

    __tablename__ = "BtcPriceTickRecord"

    id: Optional[int] = Field(default=None, primary_key=True)
    source: str
    price: float
    timestamp: datetime = Field(index=True)

    def to_tick(self) -> BtcSpotTick:
        return BtcSpotTick(source=self.source, price=self.price, timestamp=self.timestamp)


class TradeRecord(SQLModel, table=True):
    """A market opportunity the live trader observed, plus its settlement.

    The backtester re-decides each row under candidate parameters, so the row
    stores the quote at observation time (not just what was actually traded).
    """

    __tablename__ = "TradeRecord"

    id: Optional[int] = Field(default=None, primary_key=True)
    ticker: str = Field(index=True)
    strike_price: float
    entry_time: datetime = Field(index=True)
    expiration_time: datetime
    yes_bid: int                       # Cents (1-99) at entry_time
    yes_ask: int                       # Cents (1-99) at entry_time
    side: Optional[str] = None         # "yes"/"no" if the live trader traded it
    count: int = 0                     # Contracts actually filled live
    model_probability: Optional[float] = None
    volatility_used: Optional[float] = None
    settled_yes: Optional[bool] = None  # True if BTC settled above strike
    settlement_price: Optional[float] = None
    realized_pnl_cents: Optional[int] = None


class StrategyParametersRecord(SQLModel, table=True):
    """Optimizer output. The live trader reloads the row with is_active=True."""

    __tablename__ = "StrategyParametersRecord"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=_utcnow, index=True)
    min_gross_edge: float
    volatility_lookback_sec: int
    momentum_weight: float
    kelly_scaling: float
    max_trade_bankroll_pct: float
    sharpe_ratio: float
    win_rate: float
    total_net_pnl: float               # Dollars
    num_trades: int
    is_active: bool = Field(default=True, index=True)

    def to_params(self) -> StrategyParameters:
        return StrategyParameters(
            min_gross_edge=self.min_gross_edge,
            volatility_lookback_sec=self.volatility_lookback_sec,
            momentum_weight=self.momentum_weight,
            kelly_scaling=self.kelly_scaling,
            max_trade_bankroll_pct=self.max_trade_bankroll_pct,
        )
