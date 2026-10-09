"""Black-Scholes digital (cash-or-nothing) option pricing for KXBTC15M markets.

A Kalshi "BTC above K at T" contract pays $1 if BTC_T > K. Under GBM with zero
rates/drift, the risk-neutral probability is:

    P(BTC_T > K) = N(d2),   d2 = (ln(S/K) - (sigma^2 / 2) * T) / (sigma * sqrt(T))

with T in years. BTC trades 24/7, so a year is 365 * 24 * 3600 seconds.
"""

import math
from statistics import NormalDist

from core.schemas import Btc15mMarketState, FairValueOutput, MarketQuote, StrategyParameters

SECONDS_PER_YEAR: float = 365.0 * 24.0 * 3600.0

PROB_FLOOR: float = 0.01
PROB_CEIL: float = 0.99

# d2 magnitude used when the outcome is effectively decided (T == 0 or sigma == 0).
# N(8) == 1 - 6e-16, so this is "certain" while staying JSON-serializable.
D2_DEGENERATE: float = 8.0

# Momentum is normalized to a z-score (1m return / 1m expected stdev) and clipped
# so a single spike can't dominate the signal.
MOMENTUM_Z_CLIP: float = 3.0
DEFAULT_OBI_WEIGHT: float = 0.10
# Hard cap on the total microstructure shift applied to d2.
MAX_D2_SHIFT: float = 0.5

_STD_NORMAL = NormalDist()


def seconds_to_years(seconds: float) -> float:
    return seconds / SECONDS_PER_YEAR


def norm_cdf(x: float) -> float:
    return _STD_NORMAL.cdf(x)


def norm_pdf(x: float) -> float:
    return _STD_NORMAL.pdf(x)


def _validate_inputs(spot: float, strike: float, sigma: float, t_years: float) -> None:
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError(f"spot and strike must be positive (spot={spot}, strike={strike})")
    if sigma < 0.0:
        raise ValueError(f"volatility must be non-negative (sigma={sigma})")
    if t_years < 0.0:
        raise ValueError(f"time to expiry must be non-negative (t_years={t_years})")


def compute_d2(spot: float, strike: float, sigma: float, t_years: float) -> float:
    """Raw Black-Scholes d2 with r = 0.

    When T == 0 or sigma == 0 there is no diffusion left, so the outcome is the
    intrinsic one: returns +/-D2_DEGENERATE (or 0.0 exactly at the strike).
    """
    _validate_inputs(spot, strike, sigma, t_years)
    sigma_sqrt_t = sigma * math.sqrt(t_years)
    if sigma_sqrt_t == 0.0:
        if spot > strike:
            return D2_DEGENERATE
        if spot < strike:
            return -D2_DEGENERATE
        return 0.0
    return (math.log(spot / strike) - 0.5 * sigma * sigma * t_years) / sigma_sqrt_t


def microstructure_d2_shift(
    momentum_1m_pct: float,
    order_book_imbalance: float,
    sigma: float,
    momentum_weight: float,
    obi_weight: float = DEFAULT_OBI_WEIGHT,
) -> float:
    """Additive d2 shift from 1-minute momentum and order book imbalance.

    Momentum (a % return) is converted to a z-score against the 1-minute stdev
    implied by sigma, so the same raw move counts for more in a calm market than
    in a volatile one. OBI is already in [-1, 1] and is used directly. Positive
    values push probability up (bullish), negative values push it down.
    """
    momentum_z = 0.0
    sigma_1m = sigma * math.sqrt(seconds_to_years(60.0))
    if sigma_1m > 0.0:
        momentum_z = (momentum_1m_pct / 100.0) / sigma_1m
        momentum_z = max(-MOMENTUM_Z_CLIP, min(MOMENTUM_Z_CLIP, momentum_z))

    obi = max(-1.0, min(1.0, order_book_imbalance))
    shift = momentum_weight * momentum_z + obi_weight * obi
    return max(-MAX_D2_SHIFT, min(MAX_D2_SHIFT, shift))


def clamp_probability(p: float) -> float:
    return max(PROB_FLOOR, min(PROB_CEIL, p))


def probability_to_cents(p: float) -> int:
    return max(1, min(99, round(p * 100.0)))


def price_digital_option(
    state: Btc15mMarketState,
    quote: MarketQuote,
    params: StrategyParameters | None = None,
    obi_weight: float = DEFAULT_OBI_WEIGHT,
) -> FairValueOutput:
    """Price a KXBTC15M contract and compute its edge against the Kalshi YES ask."""
    if quote.ticker != state.ticker:
        raise ValueError(f"quote ticker {quote.ticker!r} does not match state ticker {state.ticker!r}")
    params = params or StrategyParameters()

    sigma = state.historical_volatility
    t_years = seconds_to_years(max(0, state.seconds_remaining))

    d2 = compute_d2(state.current_spot_price, state.strike_price, sigma, t_years)
    d2 += microstructure_d2_shift(
        momentum_1m_pct=state.momentum_1m,
        order_book_imbalance=state.order_book_imbalance,
        sigma=sigma,
        momentum_weight=params.momentum_weight,
        obi_weight=obi_weight,
    )

    model_probability = clamp_probability(norm_cdf(d2))
    return FairValueOutput(
        ticker=state.ticker,
        strike_price=state.strike_price,
        model_probability=model_probability,
        fair_cents=probability_to_cents(model_probability),
        gross_edge=model_probability - quote.yes_ask / 100.0,
        volatility_used=sigma,
        d2_score=d2,
    )
