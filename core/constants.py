"""Shared constants for the KXBTC15M trading agent."""

from typing import Final

# --- Kalshi -----------------------------------------------------------------
KALSHI_BTC_15M_SERIES: Final[str] = "KXBTC15M"
MARKET_WINDOW_SEC: Final[int] = 15 * 60

# --- Time / annualization ---------------------------------------------------
# BTC trades 24/7, so a year is the full calendar year of seconds.
SECONDS_PER_YEAR: Final[int] = 365 * 24 * 60 * 60

# --- Spot feed rolling windows ----------------------------------------------
SPOT_BUFFER_WINDOW_SEC: Final[int] = 15 * 60  # 1-second ticks retained in memory
VOLATILITY_WINDOW_SEC: Final[int] = 5 * 60
MOMENTUM_WINDOW_SEC: Final[int] = 60
# Minimum number of 1-second log returns before volatility is reported.
MIN_VOLATILITY_SAMPLES: Final[int] = 30
# Spot data older than this is considered stale and unusable for pricing.
SPOT_STALE_AFTER_SEC: Final[float] = 10.0

# --- Public spot WebSocket endpoints (no API key required) -------------------
COINBASE_WS_URL: Final[str] = "wss://ws-feed.exchange.coinbase.com"
COINBASE_PRODUCT_ID: Final[str] = "BTC-USD"
BINANCE_WS_URL: Final[str] = "wss://stream.binance.us:9443/ws/btcusd@trade"

SPOT_SOURCE_COINBASE: Final[str] = "coinbase"
SPOT_SOURCE_BINANCE: Final[str] = "binance"

# --- WebSocket reconnect backoff --------------------------------------------
WS_RECONNECT_INITIAL_SEC: Final[float] = 1.0
WS_RECONNECT_MAX_SEC: Final[float] = 30.0

# --- Persistence ------------------------------------------------------------
DEFAULT_DATABASE_URL: Final[str] = "sqlite+aiosqlite:///./btc_agent.db"
