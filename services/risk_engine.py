"""Position sizing, fee accounting, bankroll tracking, and the daily kill switch.

All money is tracked in integer cents to match Kalshi's quoting.
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

from core.schemas import FairValueOutput, OrderRequest, StrategyParameters

KALSHI_TAKER_FEE_RATE = 0.07
DAILY_DRAWDOWN_LIMIT = 0.15


def kalshi_taker_fee_cents(count: int, price_cents: int) -> int:
    """Kalshi taker fee: ceil(0.07 * C * P * (1 - P)) dollars, rounded up to the cent.

    Computed in integers so the ceiling is exact: 0.07 * C * (p/100) * (1 - p/100)
    dollars == 7 * C * p * (100 - p) / 10_000 cents.
    """
    if count <= 0:
        return 0
    numerator = 7 * count * price_cents * (100 - price_cents)
    return -(-numerator // 10_000)


def kalshi_maker_fee_cents(count: int, price_cents: int, fee_type: str | None) -> int:
    """Fee on a resting order that gets filled. Kalshi charges makers only on series whose
    `fee_type` is "quadratic_with_maker_fees": ceil(0.0175 * C * P * (1 - P)) dollars."""
    if count <= 0 or fee_type != "quadratic_with_maker_fees":
        return 0
    numerator = 175 * count * price_cents * (100 - price_cents)
    return -(-numerator // 1_000_000)


def kelly_fraction(win_probability: float, price_cents: int) -> float:
    """Full-Kelly bankroll fraction for buying a $1 binary contract, net of taker fees.

    With all-in cost c per contract (price + marginal fee), a win nets (1 - c) and
    a loss costs c, so f* = (p - c) / (1 - c).
    """
    price = price_cents / 100
    all_in_cost = price + KALSHI_TAKER_FEE_RATE * price * (1 - price)
    if all_in_cost >= 1.0 or win_probability <= all_in_cost:
        return 0.0
    return (win_probability - all_in_cost) / (1 - all_in_cost)


@dataclass(frozen=True)
class Position:
    ticker: str
    side: str
    count: int
    price_cents: int
    fee_cents: int
    opened_at: datetime

    @property
    def cost_cents(self) -> int:
        return self.count * self.price_cents


class RiskEngine:
    """Sizes orders with fractional Kelly and halts trading after a 15% daily drawdown.

    Equity is cash plus open positions marked at cost, so the drawdown check reacts
    to fees and settled losses; open positions resolve within 15 minutes.

    Optional hard caps bound real-money exposure regardless of what Kelly asks for: the
    all-in cost (price + fee) of any one order, and the all-in cost of everything bought in
    one UTC day.
    """

    def __init__(
        self,
        starting_bankroll_cents: int,
        daily_drawdown_limit: Optional[float] = DAILY_DRAWDOWN_LIMIT,  # None: never halt (paper)
        now: Optional[datetime] = None,
        max_order_cost_cents: Optional[int] = None,
        daily_spend_limit_cents: Optional[int] = None,
    ) -> None:
        self.cash_cents = starting_bankroll_cents
        self.open_exposure_cents = 0
        self.realized_pnl_cents = 0
        self.fees_paid_cents = 0
        self.daily_drawdown_limit = daily_drawdown_limit
        self._day: date = _utc_date(now)
        self._day_start_equity_cents = starting_bankroll_cents
        self._halted = False
        self.max_order_cost_cents = max_order_cost_cents
        self.daily_spend_limit_cents = daily_spend_limit_cents
        self.spent_today_cents = 0

    def remaining_daily_spend_cents(self, now: Optional[datetime] = None) -> Optional[int]:
        self._roll_day(now)
        if self.daily_spend_limit_cents is None:
            return None
        return max(self.daily_spend_limit_cents - self.spent_today_cents, 0)

    @property
    def equity_cents(self) -> int:
        return self.cash_cents + self.open_exposure_cents

    @property
    def day_start_equity_cents(self) -> int:
        return self._day_start_equity_cents

    def is_trading_halted(self, now: Optional[datetime] = None) -> bool:
        self._roll_day(now)
        return self._halted

    def size_order(
        self,
        fair_value: FairValueOutput,
        side: str,
        price_cents: int,
        params: StrategyParameters,
        probability_override: Optional[float] = None,
        now: Optional[datetime] = None,
        reasoning: str = "",
    ) -> Optional[OrderRequest]:
        """Return a fee-aware fractional-Kelly order, or None if no trade is warranted.

        `probability_override` (e.g. LLMVerdict.adjusted_probability) and
        `fair_value.model_probability` are both P(YES settles).
        """
        if side not in ("yes", "no"):
            raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
        if not 1 <= price_cents <= 99 or self.is_trading_halted(now):
            return None

        p_yes = fair_value.model_probability if probability_override is None else probability_override
        p_win = p_yes if side == "yes" else 1.0 - p_yes
        fraction = min(kelly_fraction(p_win, price_cents) * params.kelly_scaling, params.max_trade_bankroll_pct)
        if fraction <= 0:
            return None

        budget_cents = int(fraction * self.equity_cents)
        budget_cents = min(budget_cents, self.cash_cents)
        if self.max_order_cost_cents is not None:
            budget_cents = min(budget_cents, self.max_order_cost_cents)
        remaining = self.remaining_daily_spend_cents(now)
        if remaining is not None:
            budget_cents = min(budget_cents, remaining)
        count = budget_cents // price_cents
        while count > 0 and count * price_cents + kalshi_taker_fee_cents(count, price_cents) > budget_cents:
            count -= 1
        if count <= 0:
            return None

        return OrderRequest(
            ticker=fair_value.ticker,
            side=side,
            count=count,
            max_cost_cents=count * price_cents + kalshi_taker_fee_cents(count, price_cents),
            kelly_fraction=fraction,
            reasoning=reasoning,
        )

    def open_position(
        self,
        order: OrderRequest,
        price_cents: int,
        now: Optional[datetime] = None,
        fee_cents: Optional[int] = None,
    ) -> Position:
        """Book a fill. `fee_cents` is the exchange-reported fee when known (live fills)."""
        self._roll_day(now)
        fee = kalshi_taker_fee_cents(order.count, price_cents) if fee_cents is None else fee_cents
        position = Position(
            ticker=order.ticker,
            side=order.side,
            count=order.count,
            price_cents=price_cents,
            fee_cents=fee,
            opened_at=now or datetime.now(timezone.utc),
        )
        self.cash_cents -= position.cost_cents + fee
        self.open_exposure_cents += position.cost_cents
        self.fees_paid_cents += fee
        self.realized_pnl_cents -= fee
        self.spent_today_cents += position.cost_cents + fee
        self._check_drawdown()
        return position

    def settle_position(self, position: Position, settled_yes: bool, now: Optional[datetime] = None) -> int:
        """Settle at $1/contract for the winning side. Returns net PnL in cents (fees included)."""
        self._roll_day(now)
        won = settled_yes == (position.side == "yes")
        payout = position.count * 100 if won else 0
        self.open_exposure_cents -= position.cost_cents
        self.cash_cents += payout
        self.realized_pnl_cents += payout - position.cost_cents
        self._check_drawdown()
        return payout - position.cost_cents - position.fee_cents

    def _roll_day(self, now: Optional[datetime]) -> None:
        today = _utc_date(now)
        if today != self._day:
            self._day = today
            self._day_start_equity_cents = self.equity_cents
            self._halted = False
            self.spent_today_cents = 0

    def _check_drawdown(self) -> None:
        if self.daily_drawdown_limit is None:
            return
        floor = self._day_start_equity_cents * (1 - self.daily_drawdown_limit)
        if self.equity_cents <= floor:
            self._halted = True


def _utc_date(now: Optional[datetime]) -> date:
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return moment.date()
