"""Live orchestration loop for the KXBTC15M trading agent.

    BTC spot WebSocket ──> BtcStateManager ──┐
    Kalshi order books ──> KalshiOrderBookScanner (Black-Scholes digital pricing)
                               │ edge > min_gross_edge
                               ▼
                 entry guardrail ─> LLM risk gate ─> RiskEngine (fee-aware Quarter-Kelly)
                               ▼
                       KalshiClient (paper or live, rate-limited)

    Settlement poller (every 3 min): settle expired markets, book net PnL into the
    RiskEngine, and re-tune StrategyParameters nightly or after every N settled trades.

Run with: python main.py   (configuration is read from the environment, see .env.example)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import signal
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import date, datetime, time, timezone
from typing import Any, Literal, Optional

import uvicorn
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import func
from sqlmodel import col, select

from core.constants import DEFAULT_DATABASE_URL, SPOT_SOURCE_COINBASE
from core.schemas import Btc15mMarketState, BtcSpotTick, FairValueOutput, LLMVerdict, MarketQuote, OrderRequest, StrategyParameters
from core.state_manager import BtcStateManager
from db.database import Database
from db.models import (
    BtcPriceHistory,
    ContractOutcome,
    EvaluationReportRecord,
    EvaluationTraceRecord,
    MarketSnapshot,
    RuntimeSetting,
    TradeRecord,
    utc_now,
)
from execution.kalshi_client import KalshiAPIError, KalshiClient, KalshiMarket, OrderResult
from execution.order_scanner import KalshiOrderBookScanner, ScanCandidate, SpotContext
from pricing.btc_digital_option import price_digital_option
from pricing.greeks import MIN_ENTRY_SECONDS, check_entry_guardrail
from services.btc_spot_feed import BtcSpotFeed
from services.agent_stats import compute_activity, compute_stats
from services.evaluation import EvaluationReport
from services.llm_evaluator import LLMEvaluator, LLMUsage
from services.optimizer import Optimizer, load_active_parameters
from services.risk_engine import Position, RiskEngine, kalshi_taker_fee_cents
from web.app import DashboardState, create_app

logger = logging.getLogger("btc15m")

Side = Literal["yes", "no"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = DEFAULT_DATABASE_URL
    spot_source: str = SPOT_SOURCE_COINBASE
    log_level: str = "INFO"

    # Trading loop
    spot_persist_interval_sec: float = 5.0
    dashboard_refresh_sec: float = 1.0
    stats_refresh_sec: float = 5.0

    # LLM spend controls. Candidates are screened with these free checks before any paid call.
    llm_cooldown_sec: float = 60.0  # min gap between LLM evaluations of the same ticker/side
    llm_max_calls_per_market: int = 4  # per 15-minute contract
    llm_reask_edge_step: float = 0.02  # after a veto, re-ask only once the edge improves this much
    llm_min_seconds_remaining: int = 60  # too little time left to act on a verdict
    max_plausible_edge: float = 0.30  # larger "edges" are almost always bad quotes, not mispricing
    llm_daily_budget_usd: float = 5.0  # hard cap; past it the agent fails closed (no LLM, no trades)

    # Real-money controls. These are defaults: changes made on the dashboard are persisted and win.
    max_trade_usd: float = 1.00  # all-in cost (price + fee) of any single order
    max_daily_trade_usd: float = 100.00  # all-in cost of everything bought per UTC day
    allow_live_trading: bool = False  # server-side guard; the dashboard cannot enable live without it
    dashboard_password: Optional[str] = None  # HTTP basic auth; required before live trading is allowed

    # Settlement, research data & evaluation
    settlement_poll_sec: float = 180.0
    snapshot_interval_sec: float = 5.0  # unconditional market snapshots for research/backtests
    evaluation_interval_sec: float = 3600.0  # hourly report-only evaluation (separate process)
    evaluation_timeout_sec: float = 1200.0
    auto_apply_params: bool = True  # apply tuned parameters only when they win on held-out trades
    auto_apply_min_trades: int = 300
    auto_apply_cooldown_hours: float = 24.0
    optimizer_min_trades: int = 30  # manual "Optimize Parameters" button on the dashboard

    # Dashboard / health checks
    web_enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TradingAgent:
    """Owns the live components and the decision pipeline between them."""

    def __init__(
        self,
        settings: Settings,
        *,
        db: Database,
        client: KalshiClient,
        feed: BtcSpotFeed,
        evaluator: Optional[LLMEvaluator],
        dashboard: Optional[DashboardState] = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.settings = settings
        self.db = db
        self.client = client
        self.feed = feed
        self.evaluator = evaluator
        self.dashboard = dashboard or DashboardState()
        self._clock = clock

        self.state_manager = BtcStateManager(feed, clock=clock)
        self.scanner = KalshiOrderBookScanner(
            client,
            self.spot_context,
            fair_value_fn=self.fair_value,
            params=self.state_manager.params,
            on_candidate=self.on_candidate,
            idle_poll_seconds=5.0,  # the next contract opens a few seconds after the last one closes
            clock=clock,
        )
        feed.add_listener(self._on_tick)

        self.risk: Optional[RiskEngine] = None  # built from balance + DB history in restore()
        self._positions: dict[int, Position] = {}  # TradeRecord.id -> open position
        self._trade_rows: dict[str, int] = {}  # ticker -> its single TradeRecord.id
        self._last_evaluated: dict[tuple[str, str], datetime] = {}
        self._vetoed_edge: dict[tuple[str, str], float] = {}  # edge at the last LLM veto
        self._llm_calls: Counter[str] = Counter()  # per ticker
        self._llm_spend_day: Optional[date] = None
        self._llm_spend_today_usd = 0.0
        self.llm_screened: Counter[str] = Counter()  # candidates rejected before a paid call, by reason
        self._initial_bankroll_cents = 0
        self.max_trade_cents = round(settings.max_trade_usd * 100)
        self.max_daily_cents = round(settings.max_daily_trade_usd * 100)
        self.exchange_index: Optional[int] = None  # Kalshi shard the series trades on (live only)
        self._inflight: set[str] = set()
        self._handlers: set[asyncio.Task[None]] = set()
        self._order_lock = asyncio.Lock()  # sizing must see the bankroll left by the previous fill
        self._optimizer_lock = asyncio.Lock()

        self._latest_tick: Optional[BtcSpotTick] = None
        self._last_dashboard_tick: Optional[datetime] = None
        self._snapshots: dict[str, MarketSnapshot] = {}  # latest per live contract, written every few seconds

    @property
    def params(self) -> StrategyParameters:
        return self.state_manager.params

    @property
    def engine(self) -> AsyncEngine:
        return self.db.engine

    # -- startup ---------------------------------------------------------------

    async def restore(self) -> None:
        """Rebuild bankroll, open positions, and active parameters after a (re)start."""
        await self.reload_parameters()
        now = self._clock()
        self._positions.clear()
        self._trade_rows.clear()
        if self.client.is_paper:
            self.client.reset_paper_balance()
            balance = await self.client.get_balance()
        else:
            # API orders can only use collateral already on the series' exchange shard.
            if self.exchange_index is None:
                self.exchange_index = await self.client.get_series_exchange_index(self.scanner.series_ticker)
            balance = await self.client.get_balance(exchange_index=self.exchange_index)

        async with self.db.session() as session:
            open_rows = (
                await session.exec(
                    select(TradeRecord).where(
                        col(TradeRecord.settled_yes).is_(None),
                        col(TradeRecord.side).is_not(None),
                        TradeRecord.filled_count > 0,
                        TradeRecord.is_paper == self.client.is_paper,
                    )
                )
            ).all()
            # Per-contract rows are per mode: a contract traded on paper must not block (or absorb)
            # the record of a live trade on the same contract after a switch.
            active_rows = (
                await session.exec(
                    select(TradeRecord).where(
                        col(TradeRecord.expiration_time) > now, TradeRecord.is_paper == self.client.is_paper
                    )
                )
            ).all()
            settled_pnl = sum(
                r.realized_pnl_cents or 0
                for r in (
                    await session.exec(
                        select(TradeRecord).where(
                            col(TradeRecord.realized_pnl_cents).is_not(None),
                            TradeRecord.is_paper == self.client.is_paper,
                        )
                    )
                ).all()
            )

        open_cost = sum(_fill_price(r) * r.filled_count + r.fees_cents for r in open_rows)
        # Paper balances restart from PAPER_STARTING_BALANCE_CENTS, so replay realized PnL onto
        # them; a live Kalshi balance already reflects settlements but excludes open positions.
        starting = balance + settled_pnl if self.client.is_paper else balance + open_cost
        self._initial_bankroll_cents = starting - settled_pnl
        # Spend caps protect real money only; paper keeps unrestricted Kelly sizing.
        self.risk = RiskEngine(
            starting,
            now=now,
            max_order_cost_cents=None if self.client.is_paper else self.max_trade_cents,
            daily_spend_limit_cents=None if self.client.is_paper else self.max_daily_cents,
        )
        for row in open_rows:
            assert row.id is not None and row.side is not None
            order = OrderRequest(
                ticker=row.ticker,
                side=row.side,
                count=row.filled_count,
                max_cost_cents=row.max_cost_cents,
                kelly_fraction=row.kelly_fraction,
                reasoning=row.reasoning,
            )
            self._positions[row.id] = self.risk.open_position(order, _fill_price(row), now=now)
        if self.client.is_paper:
            self.client.credit_paper_balance(self.risk.cash_cents - balance)
        self.risk.spent_today_cents = await self._spent_today_cents(now)
        self._trade_rows = {r.ticker: r.id for r in active_rows if r.id is not None}
        await self._load_llm_spend(now)

        logger.info(
            "restored %s bankroll: equity $%.2f, %d open position(s)%s",
            "PAPER" if self.client.is_paper else f"LIVE (exchange {self.exchange_index})",
            self.risk.equity_cents / 100,
            len(self._positions),
            "" if self.client.is_paper else (
                f", spent today ${self.risk.spent_today_cents / 100:.2f} of ${self.max_daily_cents / 100:.2f}"
                f" (max ${self.max_trade_cents / 100:.2f}/trade)"
            ),
        )

    async def _spent_today_cents(self, now: datetime) -> int:
        midnight = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
        async with self.db.session() as session:
            rows = (
                await session.exec(
                    select(TradeRecord).where(
                        TradeRecord.is_paper == self.client.is_paper,
                        TradeRecord.filled_count > 0,
                        TradeRecord.entry_time >= midnight,
                    )
                )
            ).all()
        return sum(_fill_price(r) * r.filled_count + r.fees_cents for r in rows)

    # -- operator controls (dashboard) ------------------------------------------

    @property
    def is_live(self) -> bool:
        return not self.client.is_paper

    def live_blocked_reason(self) -> Optional[str]:
        if not self.settings.allow_live_trading:
            return "ALLOW_LIVE_TRADING is not enabled on the server"
        if not self.client.can_trade_live:
            return "Kalshi API credentials are not configured"
        if not self.settings.dashboard_password:
            return "set DASHBOARD_PASSWORD so the dashboard is locked before trading real money"
        return None

    async def load_controls(self) -> None:
        """Apply persisted dashboard overrides over the env defaults. Call before `restore()`."""
        async with self.db.session() as session:
            saved = {r.key: r.value for r in (await session.exec(select(RuntimeSetting))).all()}
        if "max_trade_usd" in saved:
            self.max_trade_cents = round(float(saved["max_trade_usd"]) * 100)
        if "max_daily_trade_usd" in saved:
            self.max_daily_cents = round(float(saved["max_daily_trade_usd"]) * 100)
        mode = saved.get("trading_mode")
        live = self.client.trading_enabled if mode is None else mode == "live"
        if live and (reason := self.live_blocked_reason()):
            logger.error("live trading requested but blocked (%s); staying in PAPER mode", reason)
            live = False
        self.client.trading_enabled = live

    async def _save_setting(self, key: str, value: str) -> None:
        async with self.db.session() as session:
            row = await session.get(RuntimeSetting, key)
            row = row or RuntimeSetting(key=key, value=value)
            row.value, row.updated_at = value, utc_now()
            session.add(row)
            await session.commit()

    async def switch_mode(self, live: bool) -> None:
        """Toggle paper/live without a restart. Open positions in the old mode still settle."""
        if live and (reason := self.live_blocked_reason()):
            raise ValueError(f"cannot enable live trading: {reason}")
        async with self._order_lock:  # never switch while an order is being sized or sent
            if live == self.is_live:
                return
            self.client.trading_enabled = live
            await self._save_setting("trading_mode", "live" if live else "paper")
            self.llm_screened.clear()  # dashboard counters are per mode
            await self.restore()
        logger.warning("switched to %s trading", "*** LIVE ***" if live else "PAPER")

    async def set_limits(self, max_trade_usd: float, max_daily_trade_usd: float) -> None:
        if not (0 < max_trade_usd <= max_daily_trade_usd):
            raise ValueError("limits must satisfy 0 < max per trade <= max per day")
        async with self._order_lock:
            self.max_trade_cents = round(max_trade_usd * 100)
            self.max_daily_cents = round(max_daily_trade_usd * 100)
            await self._save_setting("max_trade_usd", f"{max_trade_usd:.2f}")
            await self._save_setting("max_daily_trade_usd", f"{max_daily_trade_usd:.2f}")
            if self.risk is not None and self.is_live:
                self.risk.max_order_cost_cents = self.max_trade_cents
                self.risk.daily_spend_limit_cents = self.max_daily_cents
        logger.warning("spend caps set: $%.2f per trade, $%.2f per day", max_trade_usd, max_daily_trade_usd)

    def controls_state(self) -> dict[str, Any]:
        now = self._clock()
        remaining = self.risk.remaining_daily_spend_cents(now) if self.risk else None
        return {
            "mode": "live" if self.is_live else "paper",
            "live_blocked_reason": self.live_blocked_reason(),
            "max_trade_usd": self.max_trade_cents / 100,
            "max_daily_trade_usd": self.max_daily_cents / 100,
            "spent_today_usd": (self.risk.spent_today_cents / 100) if self.risk else 0.0,
            "remaining_today_usd": None if remaining is None else remaining / 100,
            "cash_usd": (self.risk.cash_cents / 100) if self.risk else None,
            "exchange_index": self.exchange_index,
        }

    async def reload_parameters(self) -> None:
        """Adopt the optimizer's active parameters (published by us or the dashboard)."""
        params = await load_active_parameters(self.engine)
        if params != self.params:
            logger.info("strategy parameters updated: %s", params.model_dump())
            await self.state_manager.update_parameters(params)
        self.scanner.params = params

    # -- spot feed -------------------------------------------------------------

    def _on_tick(self, tick: BtcSpotTick) -> None:
        self._latest_tick = tick
        last = self._last_dashboard_tick
        if last is None or (tick.timestamp - last).total_seconds() >= 1.0:
            self._last_dashboard_tick = tick.timestamp
            self.dashboard.update_spot(tick)

    def spot_context(self) -> Optional[SpotContext]:
        if self.feed.is_stale(self._clock()):
            return None
        price = self.feed.latest_price()
        vol = self.feed.volatility(self.params.volatility_lookback_sec)
        momentum = self.feed.momentum()
        if price is None or vol is None or momentum is None or vol <= 0:
            return None
        return SpotContext(spot_price=price, historical_volatility=vol, momentum_1m=momentum)

    def fair_value(self, state: Btc15mMarketState, quote: MarketQuote) -> FairValueOutput:
        return price_digital_option(state, quote, self.params)

    async def persist_spot_loop(self) -> None:
        """Downsample spot ticks into SQLite so the backtester can replay volatility."""
        last_saved: Optional[datetime] = None
        while True:
            await asyncio.sleep(self.settings.spot_persist_interval_sec)
            tick = self._latest_tick
            if tick is None or tick.timestamp == last_saved:
                continue
            try:
                async with self.db.session() as session:
                    session.add(BtcPriceHistory(source=tick.source, price=tick.price, timestamp=tick.timestamp))
                    await session.commit()
                last_saved = tick.timestamp
            except Exception:
                logger.exception("failed to persist spot tick")

    async def dashboard_loop(self) -> None:
        """Mirror scanner books into the state manager and dashboard."""
        while True:
            try:
                await self.sync_markets()
            except Exception:
                logger.exception("market sync failed")
            await asyncio.sleep(self.settings.dashboard_refresh_sec)

    async def stats_loop(self) -> None:
        while True:
            try:
                await self.refresh_stats()
            except Exception:
                logger.exception("stats refresh failed")
            await asyncio.sleep(self.settings.stats_refresh_sec)

    async def refresh_stats(self) -> None:
        assert self.risk is not None
        now = self._clock()
        async with self.db.session() as session:
            self.dashboard.stats = await compute_stats(
                session,
                now=now,
                is_paper=self.client.is_paper,
                initial_bankroll_cents=self._initial_bankroll_cents,
                equity_cents=self.risk.equity_cents,
                open_positions=len(self._positions),
                open_exposure_cents=self.risk.open_exposure_cents,
                halted=self.risk.is_trading_halted(now),
                llm_daily_budget_usd=self.settings.llm_daily_budget_usd,
                llm_spend_today_all_modes_usd=self.llm_spend_today(now),
                llm_screened=dict(self.llm_screened),
            )
            self.dashboard.activity = await compute_activity(
                session,
                now=now,
                is_paper=self.client.is_paper,
                books={t: (b.best_yes_bid, b.best_yes_ask) for t, b in self.scanner.books.items()},
            )

    async def sync_markets(self) -> None:
        for ticker in set(self.state_manager.active_tickers()) - set(self.scanner.markets):
            await self.state_manager.remove_market(ticker)
        # Per-contract LLM bookkeeping is only needed while the contract trades.
        live = set(self.scanner.markets)
        for key in [k for k in self._last_evaluated if k[0] not in live]:
            del self._last_evaluated[key]
        for key in [k for k in self._vetoed_edge if k[0] not in live]:
            del self._vetoed_edge[key]
        for ticker in [t for t in self._llm_calls if t not in live]:
            del self._llm_calls[ticker]
        await self.state_manager.prune_expired()
        for ticker, market in list(self.scanner.markets.items()):
            if market.close_time <= self._clock():
                continue
            await self.state_manager.upsert_market(ticker, market.strike_price, market.close_time)
            book = self.scanner.books.get(ticker)
            if book is None or book.best_yes_bid is None or book.best_yes_ask is None or book.is_crossed:
                continue
            quote = MarketQuote(
                ticker=ticker,
                strike_price=market.strike_price,
                yes_bid=book.best_yes_bid,
                yes_ask=book.best_yes_ask,
                volume=market.volume,
                updated_at=book.updated_at or self._clock(),
            )
            await self.state_manager.update_quote(quote)
            await self.state_manager.update_order_book_imbalance(ticker, book.imbalance)
            state = await self.state_manager.get_market_state(ticker)
            if state is not None:
                fair = self.fair_value(state, quote)
                self.dashboard.update_market(state, fair, quote)
                self._snapshots[ticker] = MarketSnapshot(
                    ts=self._clock(),
                    ticker=ticker,
                    strike_price=state.strike_price,
                    close_time=state.expiration_time,
                    seconds_remaining=state.seconds_remaining,
                    yes_bid=quote.yes_bid,
                    yes_ask=quote.yes_ask,
                    spot_price=state.current_spot_price,
                    historical_volatility=state.historical_volatility,
                    momentum_1m=state.momentum_1m,
                    order_book_imbalance=state.order_book_imbalance,
                    model_probability=fair.model_probability,
                )

    # -- decision pipeline -----------------------------------------------------

    def on_candidate(self, candidate: ScanCandidate) -> None:
        """Scanner callback. Hands off to a task so a slow LLM call never stalls the book feed."""
        # Candidates are consumed here, not via the scanner's queue; keep it from filling up.
        with contextlib.suppress(asyncio.QueueEmpty):
            self.scanner.candidates.get_nowait()
        task = asyncio.create_task(self.handle_candidate(candidate), name=f"candidate-{candidate.ticker}")
        self._handlers.add(task)
        task.add_done_callback(self._handlers.discard)

    async def handle_candidate(self, candidate: ScanCandidate) -> None:
        ticker, now = candidate.ticker, self._clock()
        if ticker in self._inflight or self._has_position(ticker):
            return
        last = self._last_evaluated.get((ticker, candidate.side))
        if last is not None and (now - last).total_seconds() < self.settings.llm_cooldown_sec:
            return
        self._inflight.add(ticker)
        try:
            await self._evaluate_and_trade(candidate)
        except Exception:
            logger.exception("candidate handling failed for %s %s", ticker, candidate.side)
        finally:
            self._inflight.discard(ticker)

    async def _evaluate_and_trade(self, c: ScanCandidate) -> None:
        assert self.risk is not None, "restore() must run before trading"
        now = self._clock()
        guard = check_entry_guardrail(c.market_state)
        if not guard.allowed:
            logger.debug("%s: %s", c.ticker, guard.reason)
            return
        if self.risk.is_trading_halted(now):
            logger.warning("daily drawdown kill switch active; ignoring %s %s", c.ticker, c.side)
            return

        screen = self._llm_screen(c, now)
        if screen is not None:
            self.llm_screened[screen] += 1
            return

        self._last_evaluated[(c.ticker, c.side)] = now
        verdict, usage = await self._llm_verdict(c, guard.greeks.delta)
        if usage is not None:
            self._llm_calls[c.ticker] += 1
            self._add_llm_spend(usage.cost_usd, now)
            if not verdict.should_bet:
                self._vetoed_edge[(c.ticker, c.side)] = c.edge
        params = self.params

        # The book may have moved while the LLM was thinking: re-read it before committing.
        now = self._clock()
        ask = self._current_ask(c.ticker, c.side)
        market = self.scanner.markets.get(c.ticker)
        p_win = verdict.adjusted_probability if c.side == "yes" else 1.0 - verdict.adjusted_probability

        order: Optional[OrderRequest] = None
        result: Optional[OrderResult] = None
        position: Optional[Position] = None
        skip_reason: Optional[str] = None
        if not verdict.should_bet:
            skip_reason = f"llm_veto: {verdict.reasoning}"
        elif ask is None or market is None:
            skip_reason = "book_empty"
        elif (market.close_time - now).total_seconds() < MIN_ENTRY_SECONDS:
            skip_reason = "entry_window_closed"
        elif p_win - ask / 100 <= params.min_gross_edge:
            skip_reason = f"edge_after_llm {p_win - ask / 100:.4f} <= {params.min_gross_edge:.4f} at {ask}c"
        else:
            async with self._order_lock:
                order = self.risk.size_order(
                    c.fair_value,
                    c.side,
                    ask,
                    params,
                    probability_override=verdict.adjusted_probability,
                    now=now,
                    reasoning=verdict.reasoning,
                )
                if order is None:
                    skip_reason = "risk_engine_no_size"
                else:
                    result, position = await self._execute(order, ask, now)
                    if position is None:
                        skip_reason = "order_not_filled"

        evaluation_id = await self._record_evaluation(c, verdict, usage, skip_reason)
        await self._record_trade(c, evaluation_id, ask, order, result, position, now)

    def _llm_screen(self, c: ScanCandidate, now: datetime) -> Optional[str]:
        """Free checks that decide whether a candidate is worth a paid LLM call. Returns the reason to skip."""
        s = self.settings
        if c.edge > s.max_plausible_edge:
            return "edge_implausible"
        if c.market_state.seconds_remaining < s.llm_min_seconds_remaining:
            return "too_close_to_expiry"
        if self._llm_calls[c.ticker] >= s.llm_max_calls_per_market:
            return "market_call_cap"
        vetoed = self._vetoed_edge.get((c.ticker, c.side))
        if vetoed is not None and c.edge < vetoed + s.llm_reask_edge_step:
            return "vetoed_no_material_change"
        if self.llm_spend_today(now) >= s.llm_daily_budget_usd:
            if self.llm_screened["daily_budget_exhausted"] == 0:
                logger.warning("LLM daily budget of $%.2f exhausted; no new trades today", s.llm_daily_budget_usd)
            return "daily_budget_exhausted"
        return None

    def llm_spend_today(self, now: datetime) -> float:
        if self._llm_spend_day != now.date():
            self._llm_spend_day, self._llm_spend_today_usd = now.date(), 0.0
            self.llm_screened.pop("daily_budget_exhausted", None)
        return self._llm_spend_today_usd

    def _add_llm_spend(self, cost_usd: float, now: datetime) -> None:
        self._llm_spend_today_usd = self.llm_spend_today(now) + cost_usd

    async def _load_llm_spend(self, now: datetime) -> None:
        midnight = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
        async with self.db.session() as session:
            spent = (
                await session.exec(
                    select(func.coalesce(func.sum(EvaluationTraceRecord.llm_cost_usd), 0.0)).where(
                        EvaluationTraceRecord.created_at >= midnight
                    )
                )
            ).one()
        self._llm_spend_day, self._llm_spend_today_usd = now.date(), float(spent)

    async def _llm_verdict(self, c: ScanCandidate, delta: float) -> tuple[LLMVerdict, Optional[LLMUsage]]:
        if self.evaluator is None:
            verdict = LLMVerdict(
                should_bet=False,
                adjusted_probability=min(max(c.fair_value.model_probability, 0.01), 0.99),
                confidence_score=0.0,
                reasoning="ANTHROPIC_API_KEY not configured; vetoing by default.",
                risk_signals=["LLM unavailable"],
            )
            return verdict, None
        assert self.risk is not None
        state = c.market_state
        context = {
            "note": "The candidate's model_probability, fair_cents and gross_edge are for the YES side. "
            "proposed_side_* fields are for the side being bought (NO ask = 100 - yes_bid).",
            "proposed_side": c.side,
            "proposed_side_ask_cents": c.ask_cents,
            "proposed_side_win_probability": round(c.model_probability, 4),
            "proposed_side_gross_edge": round(c.edge, 4),
            "min_gross_edge": self.params.min_gross_edge,
            "yes_bid": c.quote.yes_bid,
            "yes_ask": c.quote.yes_ask,
            "seconds_remaining": state.seconds_remaining,
            "spot_price": state.current_spot_price,
            "strike_price": state.strike_price,
            "spot_minus_strike": round(state.current_spot_price - state.strike_price, 2),
            "historical_volatility_annualized": round(state.historical_volatility, 4),
            "momentum_1m_pct": round(state.momentum_1m, 4),
            "order_book_imbalance": round(state.order_book_imbalance, 4),
            "digital_delta_per_usd": delta,
            "bankroll_equity_usd": self.risk.equity_cents / 100,
            "day_start_equity_usd": self.risk.day_start_equity_cents / 100,
            "open_positions": len(self._positions),
            "utc_time": self._clock().isoformat(),
        }
        return await self.evaluator.evaluate_with_usage(c.fair_value, context)

    def _current_ask(self, ticker: str, side: Side) -> Optional[int]:
        book = self.scanner.books.get(ticker)
        if book is None:
            return None
        return book.best_yes_ask if side == "yes" else book.best_no_ask

    async def _execute(
        self, order: OrderRequest, ask: int, now: datetime
    ) -> tuple[Optional[OrderResult], Optional[Position]]:
        assert self.risk is not None
        # RiskEngine's max_cost_cents includes the taker fee, but Kalshi bills fees separately and
        # KalshiClient derives the limit price as max_cost_cents // count. Limit at the ask itself.
        # Live orders are immediate-or-cancel, so an unfilled remainder never rests on the book.
        try:
            result = await self.client.submit_order(
                order.model_copy(update={"max_cost_cents": order.count * ask}),
                exchange_index=None if self.client.is_paper else self.exchange_index,
            )
        except KalshiAPIError as exc:
            logger.error("order rejected for %s %s: %s", order.ticker, order.side, exc)
            return None, None
        if result.filled_count <= 0:
            logger.info("%s %s order for %s did not fill at %dc", "PAPER" if result.paper else "LIVE", order.side, order.ticker, ask)
            return result, None

        # Book what actually executed: live fills can be cheaper than the limit, and Kalshi
        # reports the fee it charged.
        # Kalshi fills can be sub-penny (e.g. 0.2c); the RiskEngine books whole cents, so round the
        # price up (overstating cost, never understating it). The exact price is kept on the record.
        price = ask if result.avg_fill_price_cents is None else max(1, math.ceil(result.avg_fill_price_cents - 1e-9))
        fee = result.fees_cents if result.fees_cents is not None else kalshi_taker_fee_cents(result.filled_count, price)
        filled = order.model_copy(
            update={"count": result.filled_count, "max_cost_cents": result.filled_count * price + fee}
        )
        position = self.risk.open_position(filled, price, now=now, fee_cents=fee)
        logger.info(
            "%s %s %dx %s @ %dc (fee %dc, kelly %.4f, spent today $%.2f of $%.2f)",
            "PAPER" if result.paper else "LIVE",
            order.side.upper(),
            position.count,
            order.ticker,
            price,
            position.fee_cents,
            order.kelly_fraction,
            self.risk.spent_today_cents / 100,
            self.max_daily_cents / 100,
        )
        return result, position

    def _has_position(self, ticker: str) -> bool:
        return any(p.ticker == ticker for p in self._positions.values())

    async def _record_evaluation(
        self, c: ScanCandidate, verdict: LLMVerdict, usage: Optional[LLMUsage], skip_reason: Optional[str]
    ) -> int:
        s, fv = c.market_state, c.fair_value
        record = EvaluationTraceRecord(
            created_at=self._clock(),
            ticker=c.ticker,
            strike_price=s.strike_price,
            expiration_time=s.expiration_time,
            seconds_remaining=s.seconds_remaining,
            current_spot_price=s.current_spot_price,
            historical_volatility=s.historical_volatility,
            momentum_1m=s.momentum_1m,
            order_book_imbalance=s.order_book_imbalance,
            yes_bid=c.quote.yes_bid,
            yes_ask=c.quote.yes_ask,
            proposed_side=c.side,
            proposed_ask_cents=c.ask_cents,
            is_paper=self.client.is_paper,
            model_probability=fv.model_probability,
            fair_cents=fv.fair_cents,
            gross_edge=c.edge,
            volatility_used=fv.volatility_used,
            d2_score=fv.d2_score,
            llm_should_bet=verdict.should_bet,
            llm_adjusted_probability=verdict.adjusted_probability,
            llm_confidence_score=verdict.confidence_score,
            llm_reasoning=verdict.reasoning,
            llm_risk_signals=verdict.risk_signals,
            llm_model=usage.model if usage else None,
            llm_input_tokens=usage.input_tokens if usage else None,
            llm_output_tokens=usage.output_tokens if usage else None,
            llm_cache_read_tokens=usage.cache_read_tokens if usage else None,
            llm_cache_write_tokens=usage.cache_write_tokens if usage else None,
            llm_cost_usd=usage.cost_usd if usage else None,
            decision="skip" if skip_reason else "trade",
            skip_reason=skip_reason,
        )
        async with self.db.session() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)
        assert record.id is not None
        return record.id

    async def _record_trade(
        self,
        c: ScanCandidate,
        evaluation_id: int,
        ask: Optional[int],
        order: Optional[OrderRequest],
        result: Optional[OrderResult],
        position: Optional[Position],
        now: datetime,
    ) -> None:
        """Keep one TradeRecord per ticker: the backtester replays each row as one opportunity."""
        book = self.scanner.books.get(c.ticker)
        yes_bid = book.best_yes_bid if book and book.best_yes_bid is not None else c.quote.yes_bid
        yes_ask = book.best_yes_ask if book and book.best_yes_ask is not None else c.quote.yes_ask

        async with self.db.session() as session:
            row_id = self._trade_rows.get(c.ticker)
            row = await session.get(TradeRecord, row_id) if row_id is not None else None
            if row is None:
                row = TradeRecord(ticker=c.ticker, status="skipped", is_paper=self.client.is_paper)
            elif row.side is not None and row.filled_count > 0:
                return  # already traded this contract; never overwrite a live position's row
            row.evaluation_id = evaluation_id
            row.strike_price = c.market_state.strike_price
            row.expiration_time = c.market_state.expiration_time
            row.entry_time = now
            row.yes_bid, row.yes_ask = yes_bid, yes_ask
            row.model_probability = c.fair_value.model_probability
            row.volatility_used = c.fair_value.volatility_used
            if order is not None and result is not None:
                row.side = order.side
                row.count = order.count
                row.max_cost_cents = order.max_cost_cents
                row.kelly_fraction = order.kelly_fraction
                row.reasoning = order.reasoning
                row.kalshi_order_id = result.order_id
                row.is_paper = result.paper
                row.filled_count = position.count if position else 0
                row.avg_fill_price_cents = (
                    (result.avg_fill_price_cents or float(position.price_cents)) if position else None
                )
                row.fees_cents = position.fee_cents if position else 0
                row.status = _order_status(order.count, row.filled_count)
            session.add(row)
            await session.commit()
            await session.refresh(row)
        assert row.id is not None
        self._trade_rows[c.ticker] = row.id
        if position is not None:
            self._positions[row.id] = position

    # -- settlement & auto-tuning ---------------------------------------------

    async def settlement_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.settlement_poll_sec)
            try:
                await self.poll_once()
            except Exception:
                logger.exception("settlement poll failed")

    async def poll_once(self) -> None:
        await self.settle_completed()
        await self.record_contract_outcomes()
        # Parameters change only via the hourly held-out evaluation or the dashboard's optimizer.
        await self.reload_parameters()

    # -- research data & evaluation ---------------------------------------------

    async def snapshot_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.snapshot_interval_sec)
            try:
                await self.write_snapshots()
            except Exception:
                logger.exception("failed to write market snapshots")

    async def write_snapshots(self) -> int:
        """Persist the latest sample of each live contract (skips stale ones)."""
        now = self._clock()
        fresh = [
            s for s in self._snapshots.values()
            if (now - s.ts).total_seconds() <= 2 * self.settings.snapshot_interval_sec and s.close_time > now
        ]
        self._snapshots.clear()
        if fresh:
            async with self.db.session() as session:
                session.add_all(fresh)
                await session.commit()
        return len(fresh)

    async def record_contract_outcomes(self, limit: int = 20) -> int:
        """Store the result of every contract we snapshotted, traded or not."""
        now = self._clock()
        async with self.db.session() as session:
            known = select(ContractOutcome.ticker)
            pending = (
                await session.exec(
                    select(MarketSnapshot.ticker)
                    .where(MarketSnapshot.close_time <= now, col(MarketSnapshot.ticker).not_in(known))
                    .distinct()
                    .limit(limit)
                )
            ).all()
        recorded = 0
        for ticker in pending:
            try:
                market = await self.client.get_market(ticker)
            except (KalshiAPIError, KeyError, ValueError) as exc:
                logger.warning("could not fetch outcome for %s: %s", ticker, exc)
                continue
            recorded += await self._store_outcome(market)
        return recorded

    async def _store_outcome(self, market: KalshiMarket) -> int:
        if market.result not in ("yes", "no"):
            return 0
        async with self.db.session() as session:
            await session.merge(
                ContractOutcome(
                    ticker=market.ticker,
                    strike_price=market.strike_price,
                    close_time=market.close_time,
                    settled_yes=market.result == "yes",
                    settlement_price=market.settlement_price,
                )
            )
            await session.commit()
        return 1

    async def evaluation_loop(self) -> None:
        await self.load_latest_evaluation()
        while True:
            await asyncio.sleep(self.settings.evaluation_interval_sec)
            try:
                await self.run_evaluation()
            except Exception:
                logger.exception("evaluation run failed")

    async def run_evaluation(self) -> bool:
        """Run the evaluation job in a separate, low-priority process so the backtest grid can't
        starve the order-book feed. Returns True if it completed."""
        env = {
            **os.environ,
            "DATABASE_URL": self.settings.database_url,
            "AUTO_APPLY_PARAMS": "true" if self.settings.auto_apply_params else "false",
            "AUTO_APPLY_MIN_TRADES": str(self.settings.auto_apply_min_trades),
            "AUTO_APPLY_COOLDOWN_HOURS": str(self.settings.auto_apply_cooldown_hours),
        }
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "services.evaluation",
            cwd=os.path.dirname(os.path.abspath(__file__)), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(proc.communicate(), self.settings.evaluation_timeout_sec)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            logger.error("evaluation timed out after %.0fs", self.settings.evaluation_timeout_sec)
            return False
        tail = output.decode(errors="replace").strip().splitlines()[-1:] if output else []
        if proc.returncode != 0:
            logger.error("evaluation failed (exit %s): %s", proc.returncode, output.decode(errors="replace")[-2000:])
            return False
        logger.info("evaluation finished: %s", tail[0] if tail else "")
        await self.load_latest_evaluation()
        await self.reload_parameters()  # picks up parameters it applied, if any
        return True

    async def load_latest_evaluation(self) -> None:
        async with self.db.session() as session:
            row = (
                await session.exec(
                    select(EvaluationReportRecord).order_by(col(EvaluationReportRecord.created_at).desc()).limit(1)
                )
            ).first()
        if row is not None:
            self.dashboard.evaluation = EvaluationReport.model_validate_json(row.report_json)

    async def settle_completed(self) -> int:
        """Record outcomes for expired markets. Returns the number of our positions settled."""
        assert self.risk is not None
        now = self._clock()
        async with self.db.session() as session:
            rows = (
                await session.exec(
                    select(TradeRecord).where(
                        col(TradeRecord.settled_yes).is_(None), col(TradeRecord.expiration_time) <= now
                    )
                )
            ).all()
        by_ticker: dict[str, list[TradeRecord]] = {}
        for row in rows:
            by_ticker.setdefault(row.ticker, []).append(row)

        settled_positions = 0
        for ticker, group in by_ticker.items():
            try:
                market = await self.client.get_market(ticker)
            except (KalshiAPIError, KeyError, ValueError) as exc:
                logger.warning("could not fetch settlement for %s: %s", ticker, exc)
                continue
            if market.result not in ("yes", "no"):
                continue  # Kalshi has not determined the outcome yet
            await self._store_outcome(market)
            settled_yes = market.result == "yes"

            async with self.db.session() as session:
                for row in group:
                    row.settled_yes = settled_yes
                    row.settlement_price = market.settlement_price
                    row.settled_at = now
                    position = self._positions.pop(row.id, None) if row.id is not None else None
                    if position is not None:
                        won = settled_yes == (position.side == "yes")
                        row.realized_pnl_cents = self.risk.settle_position(position, settled_yes, now=now)
                        row.status = "settled"
                        self.client.credit_paper_balance(position.count * 100 if won else 0)
                        settled_positions += 1
                        logger.info("settled %s %s: net PnL %+dc", ticker, position.side, row.realized_pnl_cents)
                    elif row.side is not None and row.filled_count > 0:
                        # Opened before a paper/live switch: not on this RiskEngine's books, but its
                        # PnL still belongs in the record (Kalshi pays out live positions itself).
                        won = settled_yes == (row.side == "yes")
                        cost = round((row.avg_fill_price_cents or 0) * row.filled_count)
                        row.realized_pnl_cents = (row.filled_count * 100 if won else 0) - cost - row.fees_cents
                        row.status = "settled"
                        settled_positions += 1
                        logger.info(
                            "settled %s %s %s position: net PnL %+dc",
                            ticker, "paper" if row.is_paper else "live", row.side, row.realized_pnl_cents,
                        )
                    session.add(row)
                await session.commit()
            self._trade_rows.pop(ticker, None)

        if settled_positions:
            logger.info(
                "bankroll $%.2f, realized PnL %+dc, fees %dc%s",
                self.risk.equity_cents / 100,
                self.risk.realized_pnl_cents,
                self.risk.fees_paid_cents,
                " [HALTED]" if self.risk.is_trading_halted(now) else "",
            )
        return settled_positions

    def make_optimizer(self, engine: AsyncEngine) -> Optimizer:
        return Optimizer(
            engine,
            base_params=self.params,
            min_trades=self.settings.optimizer_min_trades,
            starting_bankroll_cents=self.risk.equity_cents if self.risk else 100_000,
        )

    async def run_optimizer(self, reason: str) -> None:
        if self._optimizer_lock.locked() or self.dashboard.optimizer_status == "running":
            return
        async with self._optimizer_lock:
            logger.info("running optimizer (%s)", reason)
            self.dashboard.optimizer_status = "running"
            try:
                result = await self.make_optimizer(self.engine).run()
            except Exception as exc:
                logger.exception("optimizer failed")
                self.dashboard.optimizer_status, self.dashboard.optimizer_error = "error", str(exc)
                return
            self.dashboard.last_optimization, self.dashboard.optimizer_status = result, "done"
            logger.info("optimizer: %s", result.message)

    # -- lifecycle -------------------------------------------------------------

    def tasks(self) -> list[asyncio.Task[None]]:
        return [
            self.feed.start(),
            asyncio.create_task(self.scanner.run(), name="kalshi-scanner"),
            asyncio.create_task(self.persist_spot_loop(), name="spot-persist"),
            asyncio.create_task(self.dashboard_loop(), name="dashboard-sync"),
            asyncio.create_task(self.stats_loop(), name="dashboard-stats"),
            asyncio.create_task(self.settlement_loop(), name="settlement-poller"),
            asyncio.create_task(self.snapshot_loop(), name="market-snapshots"),
            asyncio.create_task(self.evaluation_loop(), name="hourly-evaluation"),
        ]

    async def shutdown(self) -> None:
        self.scanner.stop()
        await self.feed.stop()
        for task in list(self._handlers):
            task.cancel()
        await asyncio.gather(*self._handlers, return_exceptions=True)


def _fill_price(row: TradeRecord) -> int:
    return round(row.avg_fill_price_cents or 0)


def _order_status(requested: int, filled: int) -> str:
    if filled <= 0:
        return "cancelled"
    return "filled" if filled >= requested else "partial"


class _Server(uvicorn.Server):
    """Uvicorn without its own signal handling; main() owns shutdown for every task."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield

    def install_signal_handlers(self) -> None:
        pass


async def run(settings: Settings) -> None:
    db = Database(settings.database_url)
    await db.init()
    client = KalshiClient.from_env()
    feed = BtcSpotFeed(settings.spot_source)
    evaluator = LLMEvaluator() if os.environ.get("ANTHROPIC_API_KEY") else None
    if evaluator is None:
        logger.error("ANTHROPIC_API_KEY is not set: every candidate will be vetoed")
    if not client.ws_auth_headers():
        logger.error("Kalshi credentials are not set: the order-book WebSocket requires them, even for paper trading")
    agent = TradingAgent(settings, db=db, client=client, feed=feed, evaluator=evaluator)
    await agent.load_controls()
    await agent.restore()
    logger.warning(
        "starting in %s mode against %s", "PAPER" if client.is_paper else "*** LIVE TRADING ***", client.base_url
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    tasks = agent.tasks()
    server: Optional[_Server] = None
    if settings.web_enabled:
        app = create_app(
            agent.dashboard,
            db.engine,
            optimizer_factory=agent.make_optimizer,
            controls=agent,
            password=settings.dashboard_password,
        )

        @app.get("/healthz")
        async def healthz() -> dict[str, Any]:
            return {
                "mode": "paper" if client.is_paper else "live",
                "spot_stale": feed.is_stale(),
                "active_markets": len(agent.scanner.markets),
                "equity_cents": agent.risk.equity_cents if agent.risk else None,
                "halted": agent.risk.is_trading_halted() if agent.risk else None,
                "max_trade_usd": agent.max_trade_cents / 100,
                "max_daily_trade_usd": agent.max_daily_cents / 100,
                "spent_today_usd": agent.risk.spent_today_cents / 100 if agent.risk else None,
            }

        server = _Server(uvicorn.Config(app, host=settings.host, port=settings.port, log_level="warning"))
        tasks.append(asyncio.create_task(server.serve(), name="web"))

    stopper = asyncio.create_task(stop.wait(), name="stop-signal")
    done, _ = await asyncio.wait([stopper, *tasks], return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        if task is not stopper and not task.cancelled() and task.exception() is not None:
            logger.error("task %s crashed", task.get_name(), exc_info=task.exception())

    logger.info("shutting down")
    if server is not None:
        server.should_exit = True
    await agent.shutdown()
    for task in tasks:
        if task.get_name() != "web":
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    stopper.cancel()
    await client.aclose()
    await db.dispose()
    if any(t is not stopper and not t.cancelled() and t.exception() for t in done):
        raise SystemExit(1)  # let the supervisor (Fly) restart us


def main() -> None:
    load_dotenv()
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run(settings))


if __name__ == "__main__":
    main()
