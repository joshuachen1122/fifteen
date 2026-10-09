import math
from datetime import datetime, timedelta, timezone

import pytest

from core.schemas import Btc15mMarketState, MarketQuote, StrategyParameters
from pricing.btc_digital_option import (
    D2_DEGENERATE,
    MAX_D2_SHIFT,
    PROB_CEIL,
    PROB_FLOOR,
    compute_d2,
    microstructure_d2_shift,
    norm_cdf,
    price_digital_option,
    seconds_to_years,
)
from pricing.greeks import MIN_ENTRY_SECONDS, check_entry_guardrail, compute_greeks, is_entry_allowed

NOW = datetime(2026, 10, 9, 14, 0, tzinfo=timezone.utc)
STRIKE = 100_000.0


def make_state(
    spot: float,
    strike: float = STRIKE,
    sigma: float = 0.6,
    seconds_remaining: int = 600,
    momentum_1m: float = 0.0,
    obi: float = 0.0,
) -> Btc15mMarketState:
    return Btc15mMarketState(
        ticker="KXBTC15M-26OCT09-B100000",
        strike_price=strike,
        expiration_time=NOW + timedelta(seconds=seconds_remaining),
        seconds_remaining=seconds_remaining,
        current_spot_price=spot,
        historical_volatility=sigma,
        momentum_1m=momentum_1m,
        order_book_imbalance=obi,
    )


def make_quote(yes_ask: int = 50) -> MarketQuote:
    return MarketQuote(
        ticker="KXBTC15M-26OCT09-B100000",
        strike_price=STRIKE,
        yes_bid=max(1, yes_ask - 2),
        yes_ask=yes_ask,
        volume=1_000,
        updated_at=NOW,
    )


def theoretical_n_d2(spot: float, strike: float, sigma: float, seconds: int) -> tuple[float, float]:
    """Independent closed form via erf, used as the reference implementation."""
    t = seconds / (365 * 24 * 3600)
    d2 = (math.log(spot / strike) - 0.5 * sigma**2 * t) / (sigma * math.sqrt(t))
    return d2, 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))


# --- Contract states against theoretical values ------------------------------

# (label, spot, sigma, seconds, expected d2, expected N(d2))
CONTRACT_STATES = [
    ("ITM", 100_200.0, 0.6, 600, 0.762127, 0.777008),
    ("OTM", 99_800.0, 0.6, 600, -0.766273, 0.221757),
    ("ATM", 100_000.0, 0.5, 900, -0.001336, 0.499467),
]


@pytest.mark.parametrize("label,spot,sigma,seconds,expected_d2,expected_p", CONTRACT_STATES)
def test_contract_state_matches_theoretical(label, spot, sigma, seconds, expected_d2, expected_p):
    ref_d2, ref_p = theoretical_n_d2(spot, STRIKE, sigma, seconds)
    assert ref_d2 == pytest.approx(expected_d2, abs=1e-6)
    assert ref_p == pytest.approx(expected_p, abs=1e-6)

    state = make_state(spot, sigma=sigma, seconds_remaining=seconds)
    out = price_digital_option(state, make_quote(yes_ask=50))

    assert out.d2_score == pytest.approx(ref_d2, abs=1e-12)
    assert out.model_probability == pytest.approx(ref_p, abs=1e-12)
    assert out.fair_cents == round(ref_p * 100)
    assert out.gross_edge == pytest.approx(ref_p - 0.50, abs=1e-12)
    assert out.volatility_used == sigma
    assert out.strike_price == STRIKE


def test_itm_above_half_otm_below_half_atm_near_half():
    itm = price_digital_option(make_state(100_200.0), make_quote())
    otm = price_digital_option(make_state(99_800.0), make_quote())
    atm = price_digital_option(make_state(100_000.0), make_quote())
    assert itm.model_probability > 0.5 > otm.model_probability
    # ATM digital sits just below 0.5 because of the -sigma^2/2 convexity term.
    assert 0.49 < atm.model_probability < 0.5


def test_d2_symmetric_in_log_moneyness_around_atm():
    # d2 is linear in ln(S/K), so +/-x moneyness is symmetric about the ATM d2 (-sigma*sqrt(T)/2).
    up, _ = theoretical_n_d2(STRIKE * math.exp(0.002), STRIKE, 0.6, 600)
    down, _ = theoretical_n_d2(STRIKE * math.exp(-0.002), STRIKE, 0.6, 600)
    assert up > 0 > down
    assert up + down == pytest.approx(2 * compute_d2(STRIKE, STRIKE, 0.6, seconds_to_years(600)))


# --- Probability bounds ------------------------------------------------------


@pytest.mark.parametrize("spot,expected", [(110_000.0, PROB_CEIL), (90_000.0, PROB_FLOOR)])
def test_probability_clamped_deep_itm_otm(spot, expected):
    out = price_digital_option(make_state(spot), make_quote())
    assert out.model_probability == expected
    assert 1 <= out.fair_cents <= 99


@pytest.mark.parametrize(
    "spot,expected_d2,expected_p",
    [(100_100.0, D2_DEGENERATE, PROB_CEIL), (99_900.0, -D2_DEGENERATE, PROB_FLOOR), (STRIKE, 0.0, 0.5)],
)
def test_expired_or_zero_vol_is_intrinsic(spot, expected_d2, expected_p):
    for state in (make_state(spot, seconds_remaining=0), make_state(spot, sigma=0.0)):
        out = price_digital_option(state, make_quote())
        assert out.d2_score == expected_d2
        assert out.model_probability == expected_p
        assert math.isfinite(out.d2_score)


@pytest.mark.parametrize(
    "spot,strike,sigma,t",
    [(0.0, STRIKE, 0.5, 0.01), (100.0, -1.0, 0.5, 0.01), (100.0, 100.0, -0.1, 0.01), (100.0, 100.0, 0.5, -1.0)],
)
def test_invalid_inputs_raise(spot, strike, sigma, t):
    with pytest.raises(ValueError):
        compute_d2(spot, strike, sigma, t)


def test_mismatched_quote_ticker_raises():
    quote = make_quote().model_copy(update={"ticker": "KXBTC15M-26OCT09-B99000"})
    with pytest.raises(ValueError):
        price_digital_option(make_state(STRIKE), quote)


# --- Momentum / OBI adjustment -----------------------------------------------


def test_no_signal_means_no_shift():
    assert microstructure_d2_shift(0.0, 0.0, sigma=0.6, momentum_weight=0.15) == 0.0


@pytest.mark.parametrize("momentum,obi", [(0.05, 0.0), (0.0, 0.5), (0.05, 0.5)])
def test_bullish_signals_raise_probability(momentum, obi):
    base = price_digital_option(make_state(STRIKE), make_quote())
    bull = price_digital_option(make_state(STRIKE, momentum_1m=momentum, obi=obi), make_quote())
    bear = price_digital_option(make_state(STRIKE, momentum_1m=-momentum, obi=-obi), make_quote())
    assert bear.model_probability < base.model_probability < bull.model_probability
    assert bull.d2_score - base.d2_score == pytest.approx(base.d2_score - bear.d2_score)


def test_momentum_shift_is_vol_normalized():
    sigma = 0.6
    sigma_1m = sigma * math.sqrt(seconds_to_years(60))
    one_sd_move_pct = sigma_1m * 100.0
    shift = microstructure_d2_shift(one_sd_move_pct, 0.0, sigma=sigma, momentum_weight=0.15)
    assert shift == pytest.approx(0.15)


def test_shift_is_capped():
    shift = microstructure_d2_shift(50.0, 1.0, sigma=0.6, momentum_weight=10.0, obi_weight=10.0)
    assert shift == MAX_D2_SHIFT
    assert microstructure_d2_shift(-50.0, -1.0, sigma=0.6, momentum_weight=10.0, obi_weight=10.0) == -MAX_D2_SHIFT


def test_momentum_weight_comes_from_strategy_params():
    state = make_state(STRIKE, momentum_1m=0.05)
    off = price_digital_option(state, make_quote(), StrategyParameters(momentum_weight=0.0), obi_weight=0.0)
    on = price_digital_option(state, make_quote(), StrategyParameters(momentum_weight=0.3), obi_weight=0.0)
    assert off.d2_score == pytest.approx(compute_d2(STRIKE, STRIKE, 0.6, seconds_to_years(600)))
    assert on.d2_score > off.d2_score


# --- Greeks ------------------------------------------------------------------


def _prob(spot: float, sigma: float, seconds: int) -> float:
    return norm_cdf(compute_d2(spot, STRIKE, sigma, seconds_to_years(seconds)))


@pytest.mark.parametrize("spot", [100_200.0, 99_800.0, 100_000.0])
def test_greeks_match_finite_differences(spot):
    sigma, seconds, h = 0.6, 600, 1.0
    g = compute_greeks(spot, STRIKE, sigma, seconds)
    fd_delta = (_prob(spot + h, sigma, seconds) - _prob(spot - h, sigma, seconds)) / (2 * h)
    fd_gamma = (
        _prob(spot + h, sigma, seconds) - 2 * _prob(spot, sigma, seconds) + _prob(spot - h, sigma, seconds)
    ) / (h * h)
    assert g.delta == pytest.approx(fd_delta, rel=1e-4)
    assert g.gamma == pytest.approx(fd_gamma, rel=1e-3, abs=1e-12)


def test_gamma_sign_flips_across_strike():
    # Digital call is convex below the strike and concave above it.
    assert compute_greeks(99_800.0, STRIKE, 0.6, 600).gamma > 0
    assert compute_greeks(100_200.0, STRIKE, 0.6, 600).gamma < 0


def test_atm_delta_explodes_as_t_goes_to_zero():
    deltas = [compute_greeks(STRIKE, STRIKE, 0.6, s).delta for s in (900, 300, 60, 10, 1)]
    assert all(a < b for a, b in zip(deltas, deltas[1:]))
    # Delta ~ 1/sqrt(T): 900s -> 1s should scale by ~sqrt(900) = 30.
    assert deltas[-1] / deltas[0] == pytest.approx(30.0, rel=1e-3)


def test_greeks_zero_at_expiry():
    g = compute_greeks(STRIKE, STRIKE, 0.6, 0)
    assert g.delta == 0.0 and g.gamma == 0.0


# --- Time-decay guardrail ----------------------------------------------------


@pytest.mark.parametrize("seconds,allowed", [(0, False), (1, False), (29, False), (30, True), (31, True), (900, True)])
def test_entry_guardrail_threshold(seconds, allowed):
    assert MIN_ENTRY_SECONDS == 30
    assert is_entry_allowed(seconds) is allowed
    result = check_entry_guardrail(make_state(STRIKE, seconds_remaining=seconds))
    assert result.allowed is allowed
    assert result.greeks.seconds_remaining == seconds
    if not allowed:
        assert "pinning" in result.reason
