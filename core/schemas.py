from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class MarketType(str, Enum):
    BTC_15M_ABOVE = "BTC_15M_ABOVE"

class BtcSpotTick(BaseModel):
    source: str          # e.g., "coinbase" or "binance"
    price: float
    timestamp: datetime

class Btc15mMarketState(BaseModel):
    ticker: str                  # e.g. "KXBTC15M-26OCT09-B82500"
    strike_price: float          # Target BTC price K
    expiration_time: datetime    # End of 15m window T
    seconds_remaining: int
    current_spot_price: float
    historical_volatility: float # 5-minute rolling annualized volatility
    momentum_1m: float          # 1-minute price return (%)
    order_book_imbalance: float  # Bid/Ask size ratio (-1.0 to 1.0)

class MarketQuote(BaseModel):
    ticker: str
    strike_price: float
    yes_bid: int                # Cents (1-99)
    yes_ask: int                # Cents (1-99)
    volume: int
    updated_at: datetime

class FairValueOutput(BaseModel):
    ticker: str
    strike_price: float
    model_probability: float    # Black-Scholes Digital Option N(d2) adjusted
    fair_cents: int             # 1-99
    gross_edge: float           # model_prob - kalshi_ask_prob
    volatility_used: float
    d2_score: float

class LLMVerdict(BaseModel):
    should_bet: bool
    adjusted_probability: float = Field(ge=0.01, le=0.99)
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    risk_signals: list[str]     # e.g., ["High volatility spike detected", "Macro news window"]

class OrderRequest(BaseModel):
    ticker: str
    side: str                  # "yes" or "no"
    count: int
    max_cost_cents: int
    kelly_fraction: float
    reasoning: str

class StrategyParameters(BaseModel):
    min_gross_edge: float = 0.035      # 3.5%
    volatility_lookback_sec: int = 300 # 5 minutes
    momentum_weight: float = 0.15
    kelly_scaling: float = 0.25        # Quarter-Kelly
    max_trade_bankroll_pct: float = 0.05