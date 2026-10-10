"""Settlement-aware pricing: KXBTC15M settles on an average, not a single print.

Per Kalshi's CRYPTO contract terms and the series rules, the expiration value is the simple
average of CF Benchmarks' BRTI over the 60 seconds before the close. Averaging removes part of
the remaining variance, and inside the final minute part of the average is already known:

    tau >= W:  A = mean of S over [tau - W, tau]       Var(ln A) ~ sigma^2 * (tau - 2W/3)
    tau <  W:  A = ((W - tau) * A_obs + tau * F) / W,   F = mean of S over [0, tau]
               YES  <=>  F >= K_eff = (W*K - (W - tau) * A_obs) / tau,  Var(ln F) ~ sigma^2 * tau / 3

(variance of the time-average of Brownian motion over [a, a + L] is a + L/3; the average is
moment-matched to a log-normal with the same mean, so P(YES) = N(d) with the usual -v/2 term.)
The spot model, by contrast, prices P(S_tau > K) with variance sigma^2 * tau.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from core.schemas import Btc15mMarketState, FairValueOutput, MarketQuote, StrategyParameters
from pricing.btc_digital_option import (
    D2_DEGENERATE,
    DEFAULT_OBI_WEIGHT,
    clamp_probability,
    microstructure_d2_shift,
    norm_cdf,
    probability_to_cents,
    seconds_to_years,
)

SETTLEMENT_WINDOW_SEC: int = 60


def settlement_average_d(
    spot: float,
    strike: float,
    sigma: float,
    seconds_remaining: float,
    observed: Sequence[float] = (),
    window_sec: int = SETTLEMENT_WINDOW_SEC,
) -> float:
    """d such that P(settlement average >= strike) = N(d), before any microstructure shift.

    `observed` holds the per-second prices already inside the averaging window (oldest first);
    it only matters once fewer than `window_sec` seconds remain. Missing seconds are filled
    with the observed mean (or spot, when nothing was observed).
    """
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError(f"spot and strike must be positive (spot={spot}, strike={strike})")
    if sigma < 0.0:
        raise ValueError(f"volatility must be non-negative (sigma={sigma})")
    tau = max(0.0, float(seconds_remaining))
    w = float(window_sec)

    if tau >= w:
        k_eff, var_seconds = strike, tau - 2.0 * w / 3.0
    else:
        elapsed = w - tau
        a_obs = (sum(observed) / len(observed)) if observed else spot
        if tau == 0.0:
            return _degenerate(a_obs, strike)
        k_eff = (w * strike - elapsed * a_obs) / tau
        var_seconds = tau / 3.0
        if k_eff <= 0.0:
            return D2_DEGENERATE  # the part already averaged guarantees YES

    v = sigma * sigma * seconds_to_years(var_seconds)
    if v <= 0.0:
        return _degenerate(spot, k_eff)
    return (math.log(spot / k_eff) - 0.5 * v) / math.sqrt(v)


def _degenerate(value: float, strike: float) -> float:
    if value > strike:
        return D2_DEGENERATE
    if value < strike:
        return -D2_DEGENERATE
    return 0.0


def price_settlement_average(
    state: Btc15mMarketState,
    quote: MarketQuote,
    params: StrategyParameters | None = None,
    observed: Sequence[float] = (),
    obi_weight: float = DEFAULT_OBI_WEIGHT,
) -> FairValueOutput:
    """Same contract as `price_digital_option`, priced against the 60-second settlement average."""
    if quote.ticker != state.ticker:
        raise ValueError(f"quote ticker {quote.ticker!r} does not match state ticker {state.ticker!r}")
    params = params or StrategyParameters()
    sigma = state.historical_volatility
    d = settlement_average_d(state.current_spot_price, state.strike_price, sigma, state.seconds_remaining, observed)
    d += microstructure_d2_shift(
        momentum_1m_pct=state.momentum_1m,
        order_book_imbalance=state.order_book_imbalance,
        sigma=sigma,
        momentum_weight=params.momentum_weight,
        obi_weight=obi_weight,
    )
    p = clamp_probability(norm_cdf(d))
    return FairValueOutput(
        ticker=state.ticker,
        strike_price=state.strike_price,
        model_probability=p,
        fair_cents=probability_to_cents(p),
        gross_edge=p - quote.yes_ask / 100.0,
        volatility_used=sigma,
        d2_score=d,
    )


def blend_with_market(fair: FairValueOutput, quote: MarketQuote, model_weight: float) -> FairValueOutput:
    """Shrink the model toward the market mid: p = w * model + (1 - w) * mid.

    The hourly evaluation fits w on settled contracts; w near 0 means the market already prices
    in what the model knows, so its "edges" are mostly model error.
    """
    if model_weight >= 1.0:
        return fair
    w = max(0.0, model_weight)
    mid = (quote.yes_bid + quote.yes_ask) / 200.0
    p = clamp_probability(w * fair.model_probability + (1.0 - w) * mid)
    return fair.model_copy(
        update={"model_probability": p, "fair_cents": probability_to_cents(p), "gross_edge": p - quote.yes_ask / 100.0}
    )
