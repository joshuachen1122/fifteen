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
import os
import signal
from collections.abc import Callable, Iterator
from datetime import date, datetime, timezone
from typing import Any, Literal, Optional

import uvicorn
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col, select

from core.constants import DEFAULT_DATABASE_URL, SPOT_SOURCE_COINBASE
from core.schemas import Btc15mMarketState, BtcSpotTick, FairValueOutput, LLMVerdict, MarketQuote, OrderRequest, StrategyParameters
from core.state_manager import BtcStateManager
from db.database import Database
from db.models import BtcPriceHistory, EvaluationTraceRecord, TradeRecord
from execution.kalshi_client import KalshiAPIError, KalshiClient, OrderResult
from execution.order_scanner import KalshiOrderBookScanner, ScanCandidate, SpotContext
from pricing.btc_digital_option import price_digital_option
from pricing.greeks import MIN_ENTRY_SECONDS, check_entry_guardrail
from services.btc_spot_feed import BtcSpotFeed
from services.llm_evaluator import LLMEvaluator
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
    llm_cooldown_sec: float = 60.0  # min gap between LLM evaluations of the same ticker/side
    spot_persist_interval_sec: float = 5.0
    dashboard_refresh_sec: float = 1.0

    # Settlement & auto-tuning
    settlement_poll_sec: float = 180.0
    optimize_every_n_settled: int = 20
    nightly_optimize_hour_utc: int = 4
    optimizer_min_trades: int = 30

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
            clock=clock,
        )
        feed.add_listener(self._on_tick)

        self.risk: Optional[RiskEngine] = None  # built from balance + DB history in restore()
        self._positions: dict[int, Position] = {}  # TradeRecord.id -> open position
        self._trade_rows: dict[str, int] = {}  # ticker -> its single TradeRecord.id
        self._last_evaluated: dict[tuple[str, str], datetime] = {}
        self._inflight: set[str] = set()
        self._handlers: set[asyncio.Task[None]] = set()
        self._order_lock = asyncio.Lock()  # sizing must see the bankroll left by the previous fill
        self._optimizer_lock = asyncio.Lock()

        self._latest_tick: Optional[BtcSpotTick] = None
        self._last_dashboard_tick: Optional[datetime] = None
        self.settled_since_optimization = 0
        self._last_nightly_run: Optional[date] = None

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
        balance = await self.client.get_balance()

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
            active_rows = (
                await session.exec(select(TradeRecord).where(col(TradeRecord.expiration_time) > now))
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
        self.risk = RiskEngine(starting, now=now)
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
        self._trade_rows = {r.ticker: r.id for r in active_rows if r.id is not None}

        logger.info(
            "restored %s bankroll: equity $%.2f, %d open position(s)",
            "PAPER" if self.client.is_paper else "LIVE",
            self.risk.equity_cents / 100,
            len(self._positions),
        )

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

    async def sync_markets(self) -> None:
        for ticker in set(self.state_manager.active_tickers()) - set(self.scanner.markets):
            await self.state_manager.remove_market(ticker)
        await self.state_manager.prune_expired()
        for ticker, market in list(self.scanner.markets.items()):
            if market.close_time <= self._clock():
                continue
            await self.state_manager.upsert_market(ticker, market.strike_price, market.close_time)
            book = self.scanner.books.get(ticker)
            if book is None or book.best_yes_bid is None or book.best_yes_ask is None:
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
                self.dashboard.update_market(state, self.fair_value(state, quote), quote)

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

        self._last_evaluated[(c.ticker, c.side)] = now
        verdict = await self._llm_verdict(c, guard.greeks.delta)
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

        evaluation_id = await self._record_evaluation(c, verdict, skip_reason)
        await self._record_trade(c, evaluation_id, ask, order, result, position, now)

    async def _llm_verdict(self, c: ScanCandidate, delta: float) -> LLMVerdict:
        if self.evaluator is None:
            return LLMVerdict(
                should_bet=False,
                adjusted_probability=min(max(c.fair_value.model_probability, 0.01), 0.99),
                confidence_score=0.0,
                reasoning="ANTHROPIC_API_KEY not configured; vetoing by default.",
                risk_signals=["LLM unavailable"],
            )
        assert self.risk is not None
        state = c.market_state
        context = {
            "proposed_side": c.side,
            "ask_cents": c.ask_cents,
            "side_win_probability": round(c.model_probability, 4),
            "gross_edge": round(c.edge, 4),
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
        return await self.evaluator.evaluate(c.fair_value, context)

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
        try:
            result = await self.client.submit_order(order.model_copy(update={"max_cost_cents": order.count * ask}))
        except KalshiAPIError as exc:
            logger.error("order rejected for %s %s: %s", order.ticker, order.side, exc)
            return None, None

        if result.filled_count < result.count and result.status == "resting":
            # Take what crossed immediately; do not leave stale resting orders on a 15m contract.
            with contextlib.suppress(KalshiAPIError):
                await self.client.cancel_order(result.order_id)
        if result.filled_count <= 0:
            return result, None

        filled = order.model_copy(
            update={
                "count": result.filled_count,
                "max_cost_cents": result.filled_count * ask + kalshi_taker_fee_cents(result.filled_count, ask),
            }
        )
        position = self.risk.open_position(filled, ask, now=now)
        logger.info(
            "%s %s %dx %s @ %dc (fee %dc, kelly %.4f)",
            "PAPER" if result.paper else "LIVE",
            order.side.upper(),
            position.count,
            order.ticker,
            ask,
            position.fee_cents,
            order.kelly_fraction,
        )
        return result, position

    def _has_position(self, ticker: str) -> bool:
        return any(p.ticker == ticker for p in self._positions.values())

    async def _record_evaluation(self, c: ScanCandidate, verdict: LLMVerdict, skip_reason: Optional[str]) -> int:
        s, fv = c.market_state, c.fair_value
        record = EvaluationTraceRecord(
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
                row.avg_fill_price_cents = float(ask) if position and ask is not None else None
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
        settled = await self.settle_completed()
        self.settled_since_optimization += settled
        now = self._clock()
        if self.settled_since_optimization >= self.settings.optimize_every_n_settled:
            await self.run_optimizer(f"{self.settled_since_optimization} trades settled")
        elif now.hour == self.settings.nightly_optimize_hour_utc and self._last_nightly_run != now.date():
            self._last_nightly_run = now.date()
            await self.run_optimizer("nightly")
        await self.reload_parameters()  # also picks up optimizer runs started from the dashboard

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
            self.settled_since_optimization = 0
            logger.info("optimizer: %s", result.message)

    # -- lifecycle -------------------------------------------------------------

    def tasks(self) -> list[asyncio.Task[None]]:
        return [
            self.feed.start(),
            asyncio.create_task(self.scanner.run(), name="kalshi-scanner"),
            asyncio.create_task(self.persist_spot_loop(), name="spot-persist"),
            asyncio.create_task(self.dashboard_loop(), name="dashboard-sync"),
            asyncio.create_task(self.settlement_loop(), name="settlement-poller"),
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
        app = create_app(agent.dashboard, db.engine, optimizer_factory=agent.make_optimizer)

        @app.get("/healthz")
        async def healthz() -> dict[str, Any]:
            return {
                "mode": "paper" if client.is_paper else "live",
                "spot_stale": feed.is_stale(),
                "active_markets": len(agent.scanner.markets),
                "equity_cents": agent.risk.equity_cents if agent.risk else None,
                "halted": agent.risk.is_trading_halted() if agent.risk else None,
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
