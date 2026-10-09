"""SQLModel persistence tables shared by the live trader, backtester, optimizer, and dashboard.

These are storage rows, not part of the `core/schemas.py` contract. Conversion
helpers map them onto the contract types where one exists.

All datetimes are stored as UTC. SQLite has no native timezone support, so
`UTCDateTime` strips tzinfo on write and re-attaches UTC on read.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Column, DateTime, TypeDecorator
from sqlmodel import Field, SQLModel

from core.schemas import BtcSpotTick, StrategyParameters


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator[datetime]):
    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime], dialect: Any) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return value.replace(tzinfo=None)

    def process_result_value(self, value: Optional[datetime], dialect: Any) -> Optional[datetime]:
        return None if value is None else value.replace(tzinfo=timezone.utc)


def _utc_field(*, default_now: bool = False, nullable: bool = False, index: bool = False) -> Any:
    if nullable:
        return Field(default=None, sa_type=UTCDateTime, nullable=True, index=index)
    if default_now:
        return Field(default_factory=utc_now, sa_type=UTCDateTime, nullable=False, index=index)
    return Field(sa_type=UTCDateTime, nullable=False, index=index)


class BtcPriceHistory(SQLModel, table=True):
    """Historical BTC spot ticks used to replay volatility & momentum."""

    __tablename__ = "btc_price_history"

    id: Optional[int] = Field(default=None, primary_key=True)
    source: str = Field(index=True)
    price: float
    timestamp: datetime = _utc_field(index=True)

    def to_tick(self) -> BtcSpotTick:
        return BtcSpotTick(source=self.source, price=self.price, timestamp=self.timestamp)


class EvaluationTraceRecord(SQLModel, table=True):
    """One full pricing + LLM evaluation pass over a single contract."""

    __tablename__ = "evaluation_traces"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = _utc_field(default_now=True, index=True)
    ticker: str = Field(index=True)

    # Market state inputs (Btc15mMarketState)
    strike_price: float
    expiration_time: datetime = _utc_field()
    seconds_remaining: int
    current_spot_price: float
    historical_volatility: float
    momentum_1m: float
    order_book_imbalance: float

    # Kalshi quote at evaluation time (MarketQuote)
    yes_bid: Optional[int] = None
    yes_ask: Optional[int] = None
    proposed_side: Optional[str] = None  # "yes" | "no"
    proposed_ask_cents: Optional[int] = None  # ask on the proposed side

    # Quant model output (FairValueOutput)
    model_probability: float
    fair_cents: int
    gross_edge: float
    volatility_used: float
    d2_score: float

    # LLM risk overlay (LLMVerdict); null when the LLM was not consulted
    llm_should_bet: Optional[bool] = None
    llm_adjusted_probability: Optional[float] = None
    llm_confidence_score: Optional[float] = None
    llm_reasoning: Optional[str] = None
    llm_risk_signals: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))

    # LLM billing; null when the LLM was not consulted or the call was not billed
    llm_model: Optional[str] = None
    llm_input_tokens: Optional[int] = None  # uncached input only
    llm_output_tokens: Optional[int] = None
    llm_cache_read_tokens: Optional[int] = None
    llm_cache_write_tokens: Optional[int] = None
    llm_cost_usd: Optional[float] = None

    is_paper: Optional[bool] = Field(default=None, index=True)  # mode at decision time; null = before modes

    # Final decision
    decision: str = Field(index=True)  # "trade" | "skip"
    skip_reason: Optional[str] = None
    strategy_parameters_id: Optional[int] = Field(default=None, foreign_key="strategy_parameters.id")


class TradeRecord(SQLModel, table=True):
    """A market opportunity the live trader observed or traded, plus its settlement.

    The backtester re-decides each row under candidate parameters, so the row
    stores the quote at observation time (not just what was actually traded).
    """

    __tablename__ = "trades"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = _utc_field(default_now=True, index=True)
    evaluation_id: Optional[int] = Field(default=None, foreign_key="evaluation_traces.id", index=True)

    # Market snapshot at entry (replayed by the backtester)
    ticker: str = Field(index=True)
    strike_price: Optional[float] = None
    entry_time: datetime = _utc_field(default_now=True, index=True)
    expiration_time: Optional[datetime] = _utc_field(nullable=True)
    yes_bid: Optional[int] = None  # Cents (1-99) at entry_time
    yes_ask: Optional[int] = None  # Cents (1-99) at entry_time
    model_probability: Optional[float] = None
    volatility_used: Optional[float] = None

    # OrderRequest; side is None for observed-but-untraded opportunities
    side: Optional[str] = None  # "yes" | "no"
    count: int = 0
    max_cost_cents: int = 0
    kelly_fraction: float = 0.0
    reasoning: str = ""

    # Exchange lifecycle
    kalshi_order_id: Optional[str] = Field(default=None, index=True)
    status: str = Field(default="pending", index=True)  # pending | skipped | filled | partial | cancelled | settled
    is_paper: bool = True
    filled_count: int = 0
    avg_fill_price_cents: Optional[float] = None
    fees_cents: int = 0

    # Settlement
    settled_at: Optional[datetime] = _utc_field(nullable=True)
    settled_yes: Optional[bool] = None  # True if BTC settled above strike
    settlement_price: Optional[float] = None
    realized_pnl_cents: Optional[int] = None  # Net of fees


class StrategyParametersRecord(SQLModel, table=True):
    """Versioned snapshot of `StrategyParameters`. The live trader reloads the row with is_active=True."""

    __tablename__ = "strategy_parameters"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = _utc_field(default_now=True, index=True)

    min_gross_edge: float
    volatility_lookback_sec: int
    momentum_weight: float
    kelly_scaling: float
    max_trade_bankroll_pct: float

    is_active: bool = Field(default=False, index=True)
    source: str = "default"  # "default" | "optimizer" | "manual"
    notes: Optional[str] = None

    # Backtest metrics, populated when source == "optimizer"
    sharpe_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    total_net_pnl: Optional[float] = None  # Dollars
    num_trades: Optional[int] = None

    @classmethod
    def from_schema(cls, params: StrategyParameters, **extra: Any) -> StrategyParametersRecord:
        return cls(**params.model_dump(), **extra)

    def to_schema(self) -> StrategyParameters:
        return StrategyParameters(**self.model_dump(include=set(StrategyParameters.model_fields)))


class RuntimeSetting(SQLModel, table=True):
    """Operator overrides set from the dashboard (trading mode, spend caps); env vars are the defaults."""

    __tablename__ = "runtime_settings"

    key: str = Field(primary_key=True)
    value: str
    updated_at: datetime = _utc_field(default_now=True)


class MarketSnapshot(SQLModel, table=True):
    """Unconditional sample of a live contract (every few seconds), whether or not it showed an edge.

    Paired with `ContractOutcome`, this is the unbiased dataset for calibration, model research
    and backtests; trades and evaluation traces only cover moments the scanner flagged.
    """

    __tablename__ = "market_snapshots"

    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime = _utc_field(index=True)
    ticker: str = Field(index=True)
    strike_price: float
    close_time: datetime = _utc_field()
    seconds_remaining: int
    yes_bid: int
    yes_ask: int
    spot_price: float
    historical_volatility: float
    momentum_1m: float
    order_book_imbalance: float
    model_probability: float  # P(YES) from the live pricing model with the parameters active at `ts`


class ContractOutcome(SQLModel, table=True):
    """Settlement of every contract we observed, traded or not."""

    __tablename__ = "contract_outcomes"

    ticker: str = Field(primary_key=True)
    strike_price: float
    close_time: datetime = _utc_field()
    settled_yes: bool
    settlement_price: Optional[float] = None
    recorded_at: datetime = _utc_field(default_now=True)


class EvaluationReportRecord(SQLModel, table=True):
    """Output of the hourly evaluation job (JSON-encoded `EvaluationReport`)."""

    __tablename__ = "evaluation_reports"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = _utc_field(default_now=True, index=True)
    report_json: str
