import math
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from types import SimpleNamespace

import anthropic
import httpx2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from core.schemas import Btc15mMarketState, BtcSpotTick, FairValueOutput, LLMVerdict, MarketQuote, OrderRequest, StrategyParameters
from db.models import BtcPriceHistory, StrategyParametersRecord, TradeRecord
from db.session import init_db
from services.backtester import SECONDS_PER_YEAR, Backtester, digital_option_probability
from services.evaluation import EvaluationReport, HoldoutBacktest, StrategySummary
from services.llm_evaluator import LLMEvaluator, LLMUsage, evaluate_btc_trade
from services.optimizer import Optimizer, ParameterGrid, load_active_parameters
from services.risk_engine import RiskEngine, kalshi_taker_fee_cents, kelly_fraction
from web.app import DashboardState, create_app

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
SMALL_GRID = ParameterGrid(
    min_gross_edge=[0.02, 0.05, 0.08],
    volatility_lookback_sec=[300, 900],
    momentum_weight=[0.0, 0.2],
)


# ---------------------------------------------------------------- fixtures


@lru_cache(maxsize=1)
def synthetic_history(days: int = 4, sigma: float = 0.5, seed: int = 7) -> tuple[tuple[BtcSpotTick, ...], tuple[TradeRecord, ...]]:
    """GBM 1-minute ticks plus one noisy-quoted near-the-money market per 15m window."""
    rng = np.random.default_rng(seed)
    minutes = days * 24 * 60 + 30
    dt = 60 / SECONDS_PER_YEAR
    log_returns = rng.normal(-0.5 * sigma**2 * dt, sigma * math.sqrt(dt), minutes)
    prices = 80_000 * np.exp(np.cumsum(log_returns))
    ticks = tuple(
        BtcSpotTick(source="coinbase", price=float(p), timestamp=T0 + timedelta(minutes=i)) for i, p in enumerate(prices)
    )

    trades = []
    for window_start in range(30, minutes - 15, 15):
        entry_idx, expiry_idx = window_start + 1, window_start + 15
        spot = float(prices[entry_idx])
        strike = round(float(prices[window_start]) / 100) * 100 + int(rng.choice([-200, -100, 0, 100, 200]))
        p_true, _ = digital_option_probability(spot, strike, sigma, 14 * 60)
        ask = int(np.clip(round(p_true * 100 + rng.normal(0, 4)) + 1, 3, 99))
        trades.append(
            TradeRecord(
                ticker=f"KXBTC15M-SIM-{window_start}-B{strike}",
                strike_price=strike,
                entry_time=ticks[entry_idx].timestamp,
                expiration_time=ticks[expiry_idx].timestamp,
                yes_bid=ask - 2,
                yes_ask=ask,
                volatility_used=sigma,
                settled_yes=bool(prices[expiry_idx] > strike),
            )
        )
    return ticks, tuple(trades)


@pytest.fixture
async def engine(tmp_path):
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}", poolclass=NullPool)
    await init_db(eng)
    yield eng
    await eng.dispose()


def fair_value(p_yes: float, ticker: str = "KXBTC15M-26OCT09-B82500", ask_cents: int = 50) -> FairValueOutput:
    return FairValueOutput(
        ticker=ticker,
        strike_price=82_500,
        model_probability=p_yes,
        fair_cents=round(p_yes * 100),
        gross_edge=p_yes - ask_cents / 100,
        volatility_used=0.45,
        d2_score=0.25,
    )


# ---------------------------------------------------------------- risk engine


def test_taker_fee_matches_kalshi_formula():
    assert kalshi_taker_fee_cents(1, 50) == 2         # 0.07 * 0.25 = $0.0175 -> 2c
    assert kalshi_taker_fee_cents(100, 50) == 175     # $1.75 exactly, no rounding
    assert kalshi_taker_fee_cents(10, 10) == 7        # $0.063 -> 7c
    assert kalshi_taker_fee_cents(0, 50) == 0


def test_quarter_kelly_sizing_is_fee_aware():
    risk = RiskEngine(starting_bankroll_cents=100_000, now=T0)
    full_kelly = kelly_fraction(0.60, 50)
    assert full_kelly == pytest.approx((0.60 - 0.5175) / (1 - 0.5175))

    order = risk.size_order(fair_value(0.60), "yes", 50, StrategyParameters(), now=T0)
    assert order is not None
    assert order.kelly_fraction == pytest.approx(full_kelly * 0.25)
    # Budget is $42.74; 83 contracts would cost 4150 + 146 fee > budget, so 82.
    assert order.count == 82
    assert order.max_cost_cents == 82 * 50 + kalshi_taker_fee_cents(82, 50) == 4244


def test_sizing_caps_at_max_trade_bankroll_pct():
    risk = RiskEngine(starting_bankroll_cents=100_000, now=T0)
    order = risk.size_order(fair_value(0.95), "yes", 50, StrategyParameters(), now=T0)
    assert order is not None
    assert order.kelly_fraction == pytest.approx(0.05)
    assert order.max_cost_cents <= 5_000


def test_no_trade_when_fees_erase_edge():
    risk = RiskEngine(starting_bankroll_cents=100_000, now=T0)
    # 1c of raw edge at 50c is smaller than the ~1.75c taker fee.
    assert risk.size_order(fair_value(0.51), "yes", 50, StrategyParameters(), now=T0) is None


def test_no_side_uses_complement_probability():
    risk = RiskEngine(starting_bankroll_cents=100_000, now=T0)
    order = risk.size_order(fair_value(0.30), "no", 60, StrategyParameters(), now=T0)
    assert order is not None and order.side == "no"
    assert order.kelly_fraction == pytest.approx(min(kelly_fraction(0.70, 60) * 0.25, 0.05))
    # LLM override flips the view: P(YES)=0.45 leaves no edge on NO at 60c.
    assert risk.size_order(fair_value(0.30), "no", 60, StrategyParameters(), probability_override=0.45, now=T0) is None


def test_daily_drawdown_kill_switch_halts_and_resets_next_day():
    risk = RiskEngine(starting_bankroll_cents=10_000, now=T0)
    order = OrderRequest(ticker="X", side="yes", count=40, max_cost_cents=0, kelly_fraction=0.1, reasoning="")
    position = risk.open_position(order, 40, now=T0)
    assert not risk.is_trading_halted(T0)

    pnl = risk.settle_position(position, settled_yes=False, now=T0 + timedelta(minutes=15))
    assert pnl == -(1600 + kalshi_taker_fee_cents(40, 40))
    assert risk.equity_cents == 10_000 + pnl                  # 8332 < 85% of 10_000
    assert risk.is_trading_halted(T0 + timedelta(minutes=16))
    assert risk.size_order(fair_value(0.9), "yes", 50, StrategyParameters(), now=T0 + timedelta(minutes=16)) is None

    next_day = T0 + timedelta(days=1, minutes=1)
    assert not risk.is_trading_halted(next_day)
    assert risk.day_start_equity_cents == risk.equity_cents


def test_drawdown_just_under_limit_keeps_trading():
    risk = RiskEngine(starting_bankroll_cents=10_000, now=T0)
    order = OrderRequest(ticker="X", side="yes", count=10, max_cost_cents=0, kelly_fraction=0.1, reasoning="")
    risk.settle_position(risk.open_position(order, 50, now=T0), settled_yes=False, now=T0)
    assert risk.equity_cents > 8_500
    assert not risk.is_trading_halted(T0)


def test_no_drawdown_limit_never_halts():
    risk = RiskEngine(starting_bankroll_cents=10_000, daily_drawdown_limit=None, now=T0)
    order = OrderRequest(ticker="X", side="yes", count=90, max_cost_cents=0, kelly_fraction=0.1, reasoning="")
    risk.settle_position(risk.open_position(order, 90, now=T0), settled_yes=False, now=T0)
    assert risk.equity_cents < 2_000 and not risk.is_trading_halted(T0)


# ---------------------------------------------------------------- pricing + backtester


def test_digital_probability_behaves():
    atm, d2 = digital_option_probability(100_000, 100_000, 0.5, 900)
    assert 0.49 < atm < 0.5 and d2 < 0                        # -sigma^2/2 drift term
    itm, _ = digital_option_probability(100_500, 100_000, 0.5, 900)
    assert itm > 0.9
    with_momentum, _ = digital_option_probability(100_000, 100_000, 0.5, 900, momentum_1m_pct=0.05, momentum_weight=0.4)
    assert with_momentum > atm
    assert digital_option_probability(200_000, 100_000, 0.5, 900)[0] == 0.99


def test_backtest_single_trade_pnl_matches_hand_calculation():
    # Flat-ish ticks one minute apart: a 60s lookback has too few returns, so the
    # recorded volatility_used is used and the probability is deterministic.
    ticks = [BtcSpotTick(source="t", price=82_700 + i, timestamp=T0 + timedelta(minutes=i)) for i in range(20)]
    trade = TradeRecord(
        ticker="KXBTC15M-T-B82500",
        strike_price=82_500,
        entry_time=T0 + timedelta(minutes=2),
        expiration_time=T0 + timedelta(minutes=15),
        yes_bid=55,
        yes_ask=57,
        volatility_used=0.40,
        settled_yes=True,
    )
    params = StrategyParameters(min_gross_edge=0.02, volatility_lookback_sec=60, momentum_weight=0.0)
    result = Backtester(ticks, [trade], starting_bankroll_cents=100_000).run(params)

    p_yes, _ = digital_option_probability(82_702, 82_500, 0.40, 13 * 60)
    assert p_yes - 0.57 >= 0.02
    risk = RiskEngine(100_000, now=trade.entry_time)
    expected = risk.size_order(fair_value(p_yes, ask_cents=57), "yes", 57, params, now=trade.entry_time)
    fee = kalshi_taker_fee_cents(expected.count, 57)
    expected_pnl = expected.count * (100 - 57) - fee

    assert result.num_trades == 1 and result.wins == 1 and result.win_rate == 1.0
    assert result.total_net_pnl == pytest.approx(expected_pnl / 100)
    assert result.total_fees == pytest.approx(fee / 100)
    assert result.final_bankroll == pytest.approx(1_000 + expected_pnl / 100)
    assert [p.equity for p in result.equity_curve] == pytest.approx([1_000, 1_000 + expected_pnl / 100])


def test_backtest_skips_trades_below_edge_threshold():
    ticks, trades = synthetic_history()
    bt = Backtester(ticks, trades)
    loose = bt.run(StrategyParameters(min_gross_edge=0.02))
    strict = bt.run(StrategyParameters(min_gross_edge=0.08))
    assert loose.num_trades > strict.num_trades
    impossible = bt.run(StrategyParameters(min_gross_edge=0.99))
    assert impossible.num_trades == 0 and impossible.total_net_pnl == 0 and impossible.sharpe_ratio == 0


def test_backtest_metrics_are_consistent():
    ticks, trades = synthetic_history()
    result = Backtester(ticks, trades).run(StrategyParameters(min_gross_edge=0.03))
    assert result.num_trades > 20
    assert 0.0 <= result.win_rate <= 1.0
    assert result.wins == round(result.win_rate * result.num_trades)
    assert result.total_net_pnl == pytest.approx(result.final_bankroll - result.starting_bankroll, abs=0.01)
    assert result.total_fees > 0
    assert len(result.equity_curve) == result.num_trades + 1
    assert math.isfinite(result.sharpe_ratio) and result.sharpe_ratio != 0
    assert 0.0 <= result.max_drawdown_pct < 1.0


def test_backtest_lookback_changes_volatility_estimate():
    ticks, trades = synthetic_history()
    bt = Backtester(ticks, trades)
    short = bt._features(120)
    long = bt._features(900)
    sigmas_short = [f.sigma for f in short if f]
    sigmas_long = [f.sigma for f in long if f]
    assert sigmas_short != sigmas_long
    # The 15-minute estimate should land near the true 50% annualized vol.
    assert 0.35 < float(np.median(sigmas_long)) < 0.65


# ---------------------------------------------------------------- optimizer


def test_default_grid_covers_requested_ranges():
    grid = ParameterGrid()
    assert min(grid.min_gross_edge) == 0.02 and max(grid.min_gross_edge) == 0.08
    assert min(grid.volatility_lookback_sec) == 60 and max(grid.volatility_lookback_sec) == 900
    assert min(grid.momentum_weight) == 0.0 and max(grid.momentum_weight) == 0.4
    assert grid.size == len(grid.candidates(StrategyParameters()))


async def test_optimizer_saves_best_params_for_live_reload(engine):
    ticks, trades = synthetic_history()
    optimizer = Optimizer(engine, grid=SMALL_GRID, min_trades=5)
    result = await optimizer.run(ticks=ticks, trades=trades)

    assert result.evaluated == SMALL_GRID.size
    assert result.best is not None and result.record_id is not None
    sharpes = [r.sharpe_ratio for r in result.leaderboard]
    assert sharpes == sorted(sharpes, reverse=True)
    assert await load_active_parameters(engine) == result.best.params

    # A second run supersedes the first: exactly one active row remains.
    await Optimizer(engine, grid=ParameterGrid(min_gross_edge=[0.04], volatility_lookback_sec=[600], momentum_weight=[0.1]), min_trades=1).run(
        ticks=ticks, trades=trades
    )
    async with AsyncSession(engine) as session:
        rows = (await session.exec(select(StrategyParametersRecord))).all()
    assert len(rows) == 2 and sum(r.is_active for r in rows) == 1
    reloaded = await load_active_parameters(engine)
    assert (reloaded.min_gross_edge, reloaded.volatility_lookback_sec, reloaded.momentum_weight) == (0.04, 600, 0.1)


async def test_optimizer_loads_history_from_sqlite(engine):
    ticks, trades = synthetic_history()
    async with AsyncSession(engine) as session:
        session.add_all(BtcPriceHistory(source=t.source, price=t.price, timestamp=t.timestamp) for t in ticks)
        session.add_all(TradeRecord(**t.model_dump()) for t in trades)
        await session.commit()

    result = await Optimizer(engine, grid=SMALL_GRID, min_trades=5).run()
    assert result.replayable_trades == len(trades)
    assert result.best is not None


async def test_optimizer_without_enough_trades_keeps_defaults(engine):
    ticks, trades = synthetic_history()
    result = await Optimizer(engine, grid=SMALL_GRID, min_trades=10_000).run(ticks=ticks, trades=trades)
    assert result.best is None and result.record_id is None
    assert await load_active_parameters(engine) == StrategyParameters()


# ---------------------------------------------------------------- LLM evaluator


class FakeMessages:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return self.response


def fake_client(response=None, exc=None):
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(response, exc)))


APPROVE = LLMVerdict(
    should_bet=True, adjusted_probability=0.61, confidence_score=0.8, reasoning="Calm regime.", risk_signals=[]
)
CONTEXT = {"realized_vol_5m": 0.42, "realized_vol_1h": 0.40, "upcoming_events": ["None scheduled"]}


async def test_llm_evaluator_returns_parsed_verdict_and_sends_risk_prompt():
    client = fake_client(SimpleNamespace(stop_reason="end_turn", parsed_output=APPROVE))
    verdict = await evaluate_btc_trade(fair_value(0.62), CONTEXT, evaluator=LLMEvaluator(client=client))
    assert verdict == APPROVE

    (call,) = client.beta.messages.calls
    assert call["model"] == "claude-sonnet-5-5"
    assert call["output_format"] is LLMVerdict
    assert call["fallbacks"] == "default"
    [system_block] = call["system"]
    assert system_block["cache_control"] == {"type": "ephemeral"}
    system = system_block["text"].lower()
    for check in ("macro news", "liquidation cascade", "volatility regime"):
        assert check in system
    user = call["messages"][0]["content"]
    assert "KXBTC15M-26OCT09-B82500" in user and "realized_vol_5m" in user


@pytest.mark.parametrize(
    "response, exc, signal",
    [
        (SimpleNamespace(stop_reason="refusal", parsed_output=None), None, "LLM refusal"),
        (SimpleNamespace(stop_reason="max_tokens", parsed_output=None), None, "LLM invalid output"),
        (SimpleNamespace(stop_reason="end_turn", parsed_output=None), None, "LLM invalid output"),
        (None, anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")), "LLM unavailable"),
        (None, ValueError("bad json"), "LLM invalid output"),
    ],
)
async def test_llm_evaluator_fails_closed(response, exc, signal):
    verdict = await LLMEvaluator(client=fake_client(response, exc)).evaluate(fair_value(0.62), CONTEXT)
    assert verdict.should_bet is False
    assert verdict.confidence_score == 0.0
    assert verdict.risk_signals == [signal]
    assert verdict.adjusted_probability == pytest.approx(0.62)


# ---------------------------------------------------------------- dashboard


def make_market(now: datetime, strike: float = 82_500, minutes_left: int = 9) -> tuple[Btc15mMarketState, FairValueOutput, MarketQuote]:
    ticker = f"KXBTC15M-26OCT09-B{int(strike)}"
    state = Btc15mMarketState(
        ticker=ticker,
        strike_price=strike,
        expiration_time=now + timedelta(minutes=minutes_left),
        seconds_remaining=minutes_left * 60,
        current_spot_price=82_640.0,
        historical_volatility=0.45,
        momentum_1m=0.031,
        order_book_imbalance=0.2,
    )
    fv = fair_value(0.664, ticker=ticker, ask_cents=60)
    quote = MarketQuote(ticker=ticker, strike_price=strike, yes_bid=58, yes_ask=60, volume=1200, updated_at=now)
    return state, fv, quote


@pytest.fixture
def dashboard(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'ui.db'}"
    engine = create_async_engine(db_url, poolclass=NullPool)
    state = DashboardState()
    now = datetime.now(timezone.utc)
    for i in range(30):
        state.update_spot(BtcSpotTick(source="coinbase", price=82_600 + i * 1.5, timestamp=now - timedelta(seconds=30 - i)))
    state.update_market(*make_market(now))
    expired_state, expired_fv, _ = make_market(now, strike=81_000, minutes_left=-1)
    state.update_market(expired_state, expired_fv)

    app = create_app(state, engine)
    with TestClient(app) as client:
        yield client, state


def test_index_renders_dark_htmx_dashboard(dashboard):
    client, _ = dashboard
    html = client.get("/").text
    assert 'data-theme="dark"' in html and "htmx.org" in html
    assert "Live BTC spot vs strike K" in html
    assert "Active 15m options" in html and "N(d₂)" in html
    assert "Parameter tuning" in html and "hourly held-out backtest" in html
    assert "/optimize" not in html and "Optimize Parameters" not in html


def test_spot_partial_shows_spot_vs_strike(dashboard):
    client, _ = dashboard
    html = client.get("/partials/spot").text
    assert "$82,643.50" in html                    # latest tick
    assert "$82,500" in html                       # nearest strike K
    assert "▲ above $144" in html
    assert "<svg" in html and "K $82,500" in html  # strike reference rule on the chart
    assert 'hx-trigger="every 2s' in html


def test_markets_partial_lists_only_active_options_with_probabilities(dashboard):
    client, _ = dashboard
    html = client.get("/partials/markets").text
    assert "KXBTC15M-26OCT09-B82500" in html
    assert "KXBTC15M-26OCT09-B81000" not in html   # expired
    assert "66.4%" in html                          # model N(d2)
    assert "58¢ / 60¢" in html
    assert "▲ Edge" in html                         # 6.4% edge >= default 3.5%


def test_tuning_panel_shows_the_hourly_held_out_backtest(dashboard):
    client, state = dashboard
    assert client.post("/optimize").status_code in (404, 405)  # tuning is automatic now
    assert "No evaluation yet" in client.get("/partials/backtest").text

    current = StrategySummary(params=StrategyParameters(), trades=40, total_net_pnl=-3.5, sharpe_ratio=-0.4, win_rate=0.45)
    best = StrategySummary(params=StrategyParameters(min_gross_edge=0.05), trades=31, total_net_pnl=12.25, sharpe_ratio=1.1, win_rate=0.55)
    state.evaluation = EvaluationReport(
        created_at=datetime(2026, 10, 10, 6, 0, tzinfo=timezone.utc), contracts=0, snapshots=0, scores=[], best_blend_weight=None,
        calibration=[], horizons=[], edges=[], llm=None, notes=[],
        backtest=HoldoutBacktest(status="better parameters found; last change < 24h ago, waiting", train_trades=210,
                                 test_trades=90, current=current, best=best),
    )
    html = client.get("/partials/backtest").text
    assert "better parameters found" in html and "2026-10-10 06:00 UTC" in html
    assert "oldest 210 replayable trades" in html and "newest 90" in html
    assert "Active parameters" in html and "Best tuned" in html and "+12.25" in html and "5.0%" in html
    assert 'hx-trigger="every 60s"' in html

async def test_evaluator_reports_usage_and_cost():
    usage = SimpleNamespace(
        input_tokens=420, output_tokens=290, cache_read_input_tokens=1170, cache_creation_input_tokens=0
    )
    response = SimpleNamespace(stop_reason="end_turn", parsed_output=APPROVE, usage=usage, model="claude-sonnet-5-5")
    verdict, billed = await LLMEvaluator(client=fake_client(response)).evaluate_with_usage(fair_value(0.62), CONTEXT)
    assert verdict == APPROVE and billed is not None
    assert billed.cost_usd == pytest.approx((420 * 2.00 + 290 * 10.00 + 1170 * 0.20) / 1_000_000)


def test_unknown_fallback_model_is_priced_conservatively():
    known = LLMUsage(model="claude-sonnet-5-5", input_tokens=1000, output_tokens=100)
    unknown = LLMUsage(model="some-fallback-model", input_tokens=1000, output_tokens=100)
    assert unknown.cost_usd > known.cost_usd


# ---------------------------------------------------------------- spend caps


def test_order_cost_cap_bounds_price_plus_fee():
    risk = RiskEngine(1_000_000, now=T0, max_order_cost_cents=100)
    order = risk.size_order(fair_value(0.70, ask_cents=47), "yes", 47, StrategyParameters(), now=T0)
    assert order is not None
    # 2 contracts @ 47c + 4c fee = 98c fits; 3 would cost 141c + fee.
    assert order.count == 2 and order.max_cost_cents == 98 <= 100


def test_daily_spend_cap_is_cumulative_and_resets_next_day():
    risk = RiskEngine(1_000_000, now=T0, max_order_cost_cents=100, daily_spend_limit_cents=150)
    params = StrategyParameters()
    first = risk.size_order(fair_value(0.70, ask_cents=47), "yes", 47, params, now=T0)
    assert first is not None
    risk.open_position(first, 47, now=T0)
    assert risk.spent_today_cents == 98 and risk.remaining_daily_spend_cents(T0) == 52

    second = risk.size_order(fair_value(0.70, ask_cents=47), "yes", 47, params, now=T0)
    assert second is not None and second.count == 1 and second.max_cost_cents <= 52
    risk.open_position(second, 47, now=T0)
    assert risk.size_order(fair_value(0.70, ask_cents=47), "yes", 47, params, now=T0) is None  # 3c left

    tomorrow = T0 + timedelta(days=1)
    assert risk.remaining_daily_spend_cents(tomorrow) == 150
    assert risk.size_order(fair_value(0.70, ask_cents=47), "yes", 47, params, now=tomorrow) is not None


def test_open_position_uses_exchange_reported_fee():
    risk = RiskEngine(10_000, now=T0)
    order = OrderRequest(ticker="KXBTC15M-T", side="no", count=2, max_cost_cents=90, kelly_fraction=0.01, reasoning="")
    position = risk.open_position(order, 43, now=T0, fee_cents=1)
    assert position.fee_cents == 1 and risk.cash_cents == 10_000 - 86 - 1 and risk.spent_today_cents == 87
