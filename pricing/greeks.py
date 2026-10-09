"""Greeks for the digital option and the near-expiry entry guardrail.

For P = N(d2 + c) (c = constant microstructure shift, 0 for pure Black-Scholes):

    Delta = dP/dS     = n(d) / (S * sigma * sqrt(T))
    Gamma = d2P/dS2   = -n(d) * (d + sigma * sqrt(T)) / (S^2 * sigma^2 * T)

where d = d2 + c. Both scale like 1/sqrt(T) and 1/T near the strike, which is
why fair value becomes unstable (pinning risk) in the final seconds.
"""

import math

from pydantic import BaseModel, ConfigDict

from core.schemas import Btc15mMarketState
from pricing.btc_digital_option import compute_d2, norm_pdf, seconds_to_years

MIN_ENTRY_SECONDS: int = 30


class DigitalGreeks(BaseModel):
    model_config = ConfigDict(frozen=True)

    delta: float  # dP/dS, probability per $1 of BTC
    gamma: float  # d2P/dS2, probability per $1^2 of BTC
    d2: float
    seconds_remaining: int


class EntryGuardrail(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: str
    greeks: DigitalGreeks


def compute_greeks(
    spot: float,
    strike: float,
    sigma: float,
    seconds_remaining: int,
    d2_shift: float = 0.0,
) -> DigitalGreeks:
    """Delta and gamma of P(BTC_T > K) with respect to spot.

    At expiry (or zero vol) the greeks are a Dirac spike at the strike and are
    reported as 0.0; callers must gate entries with `check_entry_guardrail` first.
    """
    t_years = seconds_to_years(max(0, seconds_remaining))
    d = compute_d2(spot, strike, sigma, t_years) + d2_shift
    sigma_sqrt_t = sigma * math.sqrt(t_years)
    if sigma_sqrt_t == 0.0:
        return DigitalGreeks(delta=0.0, gamma=0.0, d2=d, seconds_remaining=seconds_remaining)

    pdf = norm_pdf(d)
    delta = pdf / (spot * sigma_sqrt_t)
    gamma = -pdf * (d + sigma_sqrt_t) / (spot * spot * sigma_sqrt_t * sigma_sqrt_t)
    return DigitalGreeks(delta=delta, gamma=gamma, d2=d, seconds_remaining=seconds_remaining)


def is_entry_allowed(seconds_remaining: int) -> bool:
    return seconds_remaining >= MIN_ENTRY_SECONDS


def check_entry_guardrail(state: Btc15mMarketState, d2_shift: float = 0.0) -> EntryGuardrail:
    """Block new entries inside the final MIN_ENTRY_SECONDS of the window."""
    greeks = compute_greeks(
        spot=state.current_spot_price,
        strike=state.strike_price,
        sigma=state.historical_volatility,
        seconds_remaining=state.seconds_remaining,
        d2_shift=d2_shift,
    )
    if not is_entry_allowed(state.seconds_remaining):
        return EntryGuardrail(
            allowed=False,
            reason=(
                f"T={state.seconds_remaining}s < {MIN_ENTRY_SECONDS}s: "
                "settlement pinning risk, new entries disabled"
            ),
            greeks=greeks,
        )
    return EntryGuardrail(allowed=True, reason="ok", greeks=greeks)
