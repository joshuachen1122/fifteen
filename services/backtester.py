"""Replays observed KXBTC15M opportunities under candidate StrategyParameters.

Each TradeRecord is re-decided from scratch: volatility and momentum are rebuilt
from historical 1-minute ticks using the candidate lookback, the digital-option
probability is recomputed, and the trade is taken only if the candidate edge
threshold is met. Sizing, fees, and the daily kill switch go through RiskEngine.
"""

import heapq
import math
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import numpy as np
from pydantic import BaseModel

from core.schemas import BtcSpotTick, FairValueOutput, StrategyParameters
from db.models import TradeRecord
from services.risk_engine import Position, RiskEngine

SECONDS_PER_YEAR = 365 * 24 * 3600   # BTC trades 24/7
MIN_PROBABILITY, MAX_PROBABILITY = 0.01, 0.99


def digital_option_probability(
    spot: float,
    strike: float,
    sigma: float,
    seconds_remaining: float,
    momentum_1m_pct: float = 0.0,
    momentum_weight: float = 0.0,
) -> tuple[float, float]:
    """P(S_T > K) = N(d2) with a momentum drift tilt. Returns (probability, d2).

    The last minute's return is extrapolated over the remaining window and scaled by
    `momentum_weight` (0 = pure driftless Black-Scholes).
    """
    t_years = seconds_remaining / SECONDS_PER_YEAR
    sig_sqrt_t = sigma * math.sqrt(t_years)
    drift = momentum_weight * (momentum_1m_pct / 100.0) * (seconds_remaining / 60.0)
    d2 = (math.log(spot / strike) + drift - 0.5 * sigma**2 * t_years) / sig_sqrt_t
    probability = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
    return min(max(probability, MIN_PROBABILITY), MAX_PROBABILITY), d2


class EquityPoint(BaseModel):
    timestamp: datetime
    equity: float          # Dollars


class BacktestResult(BaseModel):
    params: StrategyParameters
    num_trades: int
    wins: int
    win_rate: float
    total_net_pnl: float   # Dollars, after fees
    total_fees: float      # Dollars
    sharpe_ratio: float    # Annualized from daily returns
    max_drawdown_pct: float
    starting_bankroll: float
    final_bankroll: float
    halted_days: int
    equity_curve: list[EquityPoint]


class _TradeFeatures(BaseModel):
    spot: float
    sigma: float
    momentum_1m_pct: float
    seconds_remaining: float


class Backtester:
    def __init__(
        self,
        ticks: Sequence[BtcSpotTick],
        trades: Sequence[TradeRecord],
        starting_bankroll_cents: int = 100_000,
    ) -> None:
        ordered = sorted(ticks, key=lambda t: t.timestamp)
        self._tick_times = [_epoch(t.timestamp) for t in ordered]
        self._tick_prices = np.array([t.price for t in ordered], dtype=float)
        self.starting_bankroll_cents = starting_bankroll_cents

        self._trades: list[tuple[TradeRecord, bool]] = []
        for trade in sorted(trades, key=lambda t: t.entry_time):
            outcome = self._settlement(trade)
            if outcome is not None:
                self._trades.append((trade, outcome))
        self._feature_cache: dict[int, list[Optional[_TradeFeatures]]] = {}
        self._probability_cache: dict[tuple[int, float], list[Optional[FairValueOutput]]] = {}

    @property
    def num_replayable_trades(self) -> int:
        return len(self._trades)

    def run(self, params: StrategyParameters) -> BacktestResult:
        fair_values = self._fair_values(params.volatility_lookback_sec, params.momentum_weight)
        first_time = self._trades[0][0].entry_time if self._trades else datetime.now(timezone.utc)
        risk = RiskEngine(self.starting_bankroll_cents, now=first_time)

        open_heap: list[tuple[float, int, Position, bool]] = []
        equity_curve = [EquityPoint(timestamp=first_time, equity=risk.equity_cents / 100)]
        daily_pnl: dict[datetime, int] = defaultdict(int)
        halted_days: set[datetime] = set()
        num_trades = wins = 0
        seq = 0

        def settle_until(cutoff: float) -> None:
            nonlocal wins
            while open_heap and open_heap[0][0] <= cutoff:
                expiry, _, position, settled_yes = heapq.heappop(open_heap)
                expiry_dt = datetime.fromtimestamp(expiry, tz=timezone.utc)
                pnl = risk.settle_position(position, settled_yes, now=expiry_dt)
                daily_pnl[_day(expiry_dt)] += pnl
                wins += pnl > 0
                equity_curve.append(EquityPoint(timestamp=expiry_dt, equity=risk.equity_cents / 100))

        for (trade, settled_yes), fair_value in zip(self._trades, fair_values):
            settle_until(_epoch(trade.entry_time))
            if fair_value is None:
                continue
            if risk.is_trading_halted(trade.entry_time):
                halted_days.add(_day(trade.entry_time))
                continue

            side, price_cents, edge = self._best_side(fair_value.model_probability, trade)
            if edge < params.min_gross_edge:
                continue
            order = risk.size_order(fair_value, side, price_cents, params, now=trade.entry_time)
            if order is None:
                continue
            # Net PnL (fees included) is booked on the settlement day by settle_until.
            position = risk.open_position(order, price_cents, now=trade.entry_time)
            heapq.heappush(open_heap, (_epoch(trade.expiration_time), seq, position, settled_yes))
            seq += 1
            num_trades += 1

        settle_until(math.inf)

        start = self.starting_bankroll_cents / 100
        final = risk.equity_cents / 100
        return BacktestResult(
            params=params,
            num_trades=num_trades,
            wins=wins,
            win_rate=wins / num_trades if num_trades else 0.0,
            total_net_pnl=round(final - start, 2),
            total_fees=risk.fees_paid_cents / 100,
            sharpe_ratio=_daily_sharpe(daily_pnl, self.starting_bankroll_cents),
            max_drawdown_pct=_max_drawdown([p.equity for p in equity_curve]),
            starting_bankroll=start,
            final_bankroll=final,
            halted_days=len(halted_days),
            equity_curve=equity_curve,
        )

    @staticmethod
    def _best_side(p_yes: float, trade: TradeRecord) -> tuple[str, int, float]:
        """Pick the side with the larger edge. Buying NO costs (100 - yes_bid) cents."""
        yes_edge = p_yes - trade.yes_ask / 100
        no_price = 100 - trade.yes_bid
        no_edge = (1 - p_yes) - no_price / 100
        if yes_edge >= no_edge:
            return "yes", trade.yes_ask, yes_edge
        return "no", no_price, no_edge

    def _fair_values(self, lookback_sec: int, momentum_weight: float) -> list[Optional[FairValueOutput]]:
        key = (lookback_sec, round(momentum_weight, 6))
        if key in self._probability_cache:
            return self._probability_cache[key]
        values: list[Optional[FairValueOutput]] = []
        for (trade, _), features in zip(self._trades, self._features(lookback_sec)):
            if features is None:
                values.append(None)
                continue
            probability, d2 = digital_option_probability(
                features.spot,
                trade.strike_price,
                features.sigma,
                features.seconds_remaining,
                features.momentum_1m_pct,
                momentum_weight,
            )
            values.append(
                FairValueOutput(
                    ticker=trade.ticker,
                    strike_price=trade.strike_price,
                    model_probability=probability,
                    fair_cents=min(max(round(probability * 100), 1), 99),
                    gross_edge=probability - trade.yes_ask / 100,
                    volatility_used=features.sigma,
                    d2_score=d2,
                )
            )
        self._probability_cache[key] = values
        return values

    def _features(self, lookback_sec: int) -> list[Optional[_TradeFeatures]]:
        if lookback_sec not in self._feature_cache:
            self._feature_cache[lookback_sec] = [self._trade_features(t, lookback_sec) for t, _ in self._trades]
        return self._feature_cache[lookback_sec]

    def _trade_features(self, trade: TradeRecord, lookback_sec: int) -> Optional[_TradeFeatures]:
        entry = _epoch(trade.entry_time)
        seconds_remaining = _epoch(trade.expiration_time) - entry
        end = bisect_right(self._tick_times, entry)
        if end == 0 or seconds_remaining <= 0:
            return None
        spot = float(self._tick_prices[end - 1])

        sigma = self._realized_vol(entry - lookback_sec, end)
        if sigma is None:
            sigma = trade.volatility_used
        if not sigma or sigma <= 0:
            return None

        prior = bisect_right(self._tick_times, entry - 60)
        momentum = (spot / float(self._tick_prices[prior - 1]) - 1) * 100 if prior > 0 else 0.0
        return _TradeFeatures(spot=spot, sigma=sigma, momentum_1m_pct=momentum, seconds_remaining=seconds_remaining)

    def _realized_vol(self, window_start: float, end: int) -> Optional[float]:
        """Annualized stdev of log returns for ticks in (window_start, entry]. Needs >= 2 returns."""
        # Include the tick at/just before window_start so the window's first return is counted.
        start = max(bisect_right(self._tick_times, window_start) - 1, 0)
        prices = self._tick_prices[start:end]
        if len(prices) < 3:
            return None
        returns = np.diff(np.log(prices))
        dt = (self._tick_times[end - 1] - self._tick_times[start]) / (len(prices) - 1)
        if dt <= 0:
            return None
        return float(np.std(returns, ddof=1) * math.sqrt(SECONDS_PER_YEAR / dt))

    def _settlement(self, trade: TradeRecord) -> Optional[bool]:
        if trade.settled_yes is not None:
            return trade.settled_yes
        if trade.settlement_price is not None:
            return trade.settlement_price > trade.strike_price
        # Fall back to the last tick within a minute before expiry.
        expiry = _epoch(trade.expiration_time)
        idx = bisect_right(self._tick_times, expiry)
        if idx == 0 or expiry - self._tick_times[idx - 1] > 60:
            return None
        return float(self._tick_prices[idx - 1]) > trade.strike_price


def _epoch(moment: datetime) -> float:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _day(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    return datetime(moment.year, moment.month, moment.day, tzinfo=timezone.utc)


def _daily_sharpe(daily_pnl: dict[datetime, int], starting_bankroll_cents: int) -> float:
    """Annualized Sharpe of daily returns (sqrt(365)), counting idle days in the span as 0%."""
    if not daily_pnl:
        return 0.0
    first, last = min(daily_pnl), max(daily_pnl)
    equity = float(starting_bankroll_cents)
    returns: list[float] = []
    day = first
    while day <= last:
        pnl = daily_pnl.get(day, 0)
        returns.append(pnl / equity if equity > 0 else 0.0)
        equity += pnl
        day += timedelta(days=1)
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(returns) / std * math.sqrt(365))


def _max_drawdown(equity: list[float]) -> float:
    peak, worst = -math.inf, 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst
