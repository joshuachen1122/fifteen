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
    # Exact LLM input ({"candidate": FairValueOutput, "context": {...}}) so decisions can be replayed
    # through other models; null for decisions recorded before replays existed.
    llm_input_json: Optional[str] = None

    is_paper: Optional[bool] = Field(default=None, index=True)  # mode at decision time; null = before modes

    # Final decision
    # False: decided on the model alone, with the LLM verdict recorded afterwards in shadow (it gated
    # nothing). Null = the LLM gated the decision (all rows before shadow mode).
    llm_gated: Optional[bool] = None
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
    """Unconditional sample of a live contract, whether or not it showed an edge: every 5 seconds,
    and every second in the final two minutes (the settlement average and the entry window).

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
    model_probability: float  # P(YES) the agent traded on at `ts` (active model, after any market blend)
    # Raw model probabilities, recorded side by side so the evaluation can compare them on the
    # same contracts; null before the settlement-aware model existed (those rows are spot-only).
    spot_model_probability: Optional[float] = None
    settlement_model_probability: Optional[float] = None
    # Coinbase-to-CF-Benchmarks gap added to spot (services/spot_calibration.py); null before the
    # correction existed, which also marks the older model generation for the evaluation.
    index_gap: Optional[float] = None
    # Top of book in contracts and resting size within 5c of the best price (fill realism).
    yes_bid_size: Optional[float] = None
    yes_ask_size: Optional[float] = None  # = size of the best NO bid
    bid_depth_5c: Optional[float] = None
    ask_depth_5c: Optional[float] = None
    # Median mid across the index exchanges we record (`IndexSample`), uncalibrated.
    composite_spot: Optional[float] = None
    # Inside the settlement window: mean of the (calibrated) spot seconds already in the average,
    # and how many seconds that covers. These are the settlement model's observed inputs.
    observed_average: Optional[float] = None
    observed_seconds: Optional[int] = None


class IndexSample(SQLModel, table=True):
    """Index-exchange order books reduced to proxies for the CF Benchmarks index
    (services/exchange_recorder.py): every 5 seconds, and every second while a contract is in its
    final two minutes (each contract's settlement average, which is also the next one's strike).

    Before 2026-10-13 ~21:00 UTC the venue mids came from ticker channels that lagged by 1-3 s and
    the depth / estimate columns are null."""

    __tablename__ = "index_samples"

    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime = _utc_field(index=True)
    coinbase: Optional[float] = None  # best bid/ask mid per venue
    kraken: Optional[float] = None
    bitstamp: Optional[float] = None
    gemini: Optional[float] = None
    cryptocom: Optional[float] = None
    bullish: Optional[float] = None
    coinbase_depth: Optional[float] = None  # mid of the VWAPs to buy and to sell 1 BTC
    kraken_depth: Optional[float] = None
    bitstamp_depth: Optional[float] = None
    gemini_depth: Optional[float] = None
    cryptocom_depth: Optional[float] = None
    bullish_depth: Optional[float] = None
    composite: Optional[float] = None  # median of the fresh venue mids
    depth_composite: Optional[float] = None  # median of the fresh venue depth mids
    brti_estimate: Optional[float] = None  # the BRTI calculation on the consolidated visible books
    sources: int = 0


class KalshiTapeBar(SQLModel, table=True):
    """Public Kalshi trades aggregated into 1-second bars over each contract's final minutes
    (a full contract prints ~30k trades, too many to keep raw)."""

    __tablename__ = "kalshi_tape_bars"

    id: Optional[int] = Field(default=None, primary_key=True)
    ticker: str = Field(index=True)
    start: datetime = _utc_field(index=True)
    seconds: int = 1
    trades: int
    volume: float  # contracts
    taker_yes_volume: float  # contracts bought by YES takers
    vwap_yes_cents: float
    low_yes_cents: float
    high_yes_cents: float


class ContractOutcome(SQLModel, table=True):
    """Settlement of every contract we observed, traded or not."""

    __tablename__ = "contract_outcomes"

    ticker: str = Field(primary_key=True)
    strike_price: float
    close_time: datetime = _utc_field()
    settled_yes: bool
    settlement_price: Optional[float] = None
    recorded_at: datetime = _utc_field(default_now=True)
    tape_bars: Optional[int] = None  # KalshiTapeBar rows stored for it; null = tape not fetched yet


class EvaluationReportRecord(SQLModel, table=True):
    """Output of the hourly evaluation job (JSON-encoded `EvaluationReport`)."""

    __tablename__ = "evaluation_reports"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = _utc_field(default_now=True, index=True)
    report_json: str


class MakerShadowOrder(SQLModel, table=True):
    """Hypothetical resting (maker) bid placed alongside an LLM-reviewed candidate. Never sent.

    Resolved after `cancel_at` against Kalshi's public trade tape: a taker selling our side at or
    below our price after we posted would have filled us. Compared with buying at the ask, this
    measures the fee + spread saved and the adverse selection paid (fills skew toward losers).
    """

    __tablename__ = "maker_shadow_orders"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = _utc_field(default_now=True, index=True)
    evaluation_id: Optional[int] = Field(default=None, foreign_key="evaluation_traces.id", index=True)
    ticker: str = Field(index=True)
    side: str  # "yes" | "no"
    is_paper: bool
    llm_should_bet: Optional[bool] = None
    close_time: datetime = _utc_field()
    cancel_at: datetime = _utc_field(index=True)
    limit_cents: int  # resting bid on our side
    joined_queue: bool  # True when the spread was 1c and we joined the best bid instead of improving it
    taker_ask_cents: int  # what buying immediately would have cost
    maker_fee_cents: int  # per contract, from the series' fee schedule
    win_probability: float  # model/LLM P(our side wins) at posting
    resolved_at: Optional[datetime] = _utc_field(nullable=True, index=True)
    filled: Optional[bool] = None
    filled_at: Optional[datetime] = _utc_field(nullable=True)
    fill_trade_cents: Optional[float] = None  # price of the first trade that would have filled us


class LLMReplayResult(SQLModel, table=True):
    """A logged LLM decision re-asked through another model (cost/quality experiment)."""

    __tablename__ = "llm_replays"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = _utc_field(default_now=True, index=True)
    trace_id: int = Field(foreign_key="evaluation_traces.id", index=True)
    model: str = Field(index=True)
    effort: str
    should_bet: bool
    adjusted_probability: float
    failed: bool = False  # fail-closed veto (API error, invalid output)
    cost_usd: Optional[float] = None
