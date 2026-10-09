"""Dark-mode HTMX dashboard: live spot vs strike, active 15m options, backtests.

Run with: uvicorn web.app:app --reload
The live trader pushes data in through `DashboardState.update_spot` / `update_market`.
"""

import asyncio
import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncEngine

from core.schemas import Btc15mMarketState, BtcSpotTick, FairValueOutput, MarketQuote, StrategyParameters
from db.session import init_db, make_engine
from services.optimizer import OptimizationResult, Optimizer, load_active_parameters
from web.charts import ReferenceLine, line_chart_svg, usd, usd_cents

logger = logging.getLogger(__name__)

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
SPOT_HISTORY_LEN = 900


class MarketSnapshot(BaseModel):
    state: Btc15mMarketState
    fair_value: Optional[FairValueOutput] = None
    quote: Optional[MarketQuote] = None


class DashboardState:
    """In-memory view model shared by the live trader and the dashboard."""

    def __init__(self) -> None:
        self.spot: Optional[BtcSpotTick] = None
        self.spot_history: deque[BtcSpotTick] = deque(maxlen=SPOT_HISTORY_LEN)
        self.markets: dict[str, MarketSnapshot] = {}
        self.optimizer_status: Literal["idle", "running", "done", "error"] = "idle"
        self.optimizer_error: Optional[str] = None
        self.last_optimization: Optional[OptimizationResult] = None
        self.optimizer_task: Optional[asyncio.Task[None]] = None

    def update_spot(self, tick: BtcSpotTick) -> None:
        self.spot = tick
        self.spot_history.append(tick)

    def update_market(
        self,
        state: Btc15mMarketState,
        fair_value: Optional[FairValueOutput] = None,
        quote: Optional[MarketQuote] = None,
    ) -> None:
        self.markets[state.ticker] = MarketSnapshot(state=state, fair_value=fair_value, quote=quote)

    def active_markets(self, now: Optional[datetime] = None) -> list[MarketSnapshot]:
        now = now or datetime.now(timezone.utc)
        live = [m for m in self.markets.values() if _aware(m.state.expiration_time) > now]
        return sorted(live, key=lambda m: (m.state.expiration_time, m.state.strike_price))

    def focus_market(self) -> Optional[MarketSnapshot]:
        """The active market whose strike is closest to spot (the most contested one)."""
        markets = self.active_markets()
        if not markets:
            return None
        spot = self.spot.price if self.spot else markets[0].state.current_spot_price
        return min(markets, key=lambda m: abs(m.state.strike_price - spot))


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def create_app(
    state: Optional[DashboardState] = None,
    engine: Optional[AsyncEngine] = None,
    optimizer_factory: Optional[Callable[[AsyncEngine], Optimizer]] = None,
) -> FastAPI:
    state = state or DashboardState()
    engine = engine or make_engine()
    optimizer_factory = optimizer_factory or (lambda e: Optimizer(e))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await init_db(engine)
        yield
        if state.optimizer_task and not state.optimizer_task.done():
            state.optimizer_task.cancel()

    app = FastAPI(title="BTC 15m Kalshi Agent", lifespan=lifespan)
    app.state.dashboard = state
    app.state.engine = engine

    async def run_optimizer() -> None:
        try:
            state.last_optimization = await optimizer_factory(engine).run()
            state.optimizer_status = "done"
        except Exception as exc:  # surfaced on the dashboard instead of dying silently
            logger.exception("Optimizer run failed")
            state.optimizer_error = str(exc)
            state.optimizer_status = "error"

    def spot_context() -> dict[str, Any]:
        focus = state.focus_market()
        spot = state.spot.price if state.spot else (focus.state.current_spot_price if focus else None)
        refs = [ReferenceLine(focus.state.strike_price, f"K {usd(focus.state.strike_price)}")] if focus else []
        chart = line_chart_svg(
            [(_aware(t.timestamp), t.price) for t in state.spot_history],
            title="BTC spot price vs strike K",
            value_format=usd,
            reference_lines=refs,
            chart_id="spot-chart",
        )
        return {
            "spot": spot,
            "spot_source": state.spot.source if state.spot else None,
            "spot_time": state.spot.timestamp if state.spot else None,
            "focus": focus,
            "distance": (spot - focus.state.strike_price) if (focus and spot is not None) else None,
            "chart": chart,
            "history": list(state.spot_history)[-15:],
        }

    async def backtest_context() -> dict[str, Any]:
        result = state.last_optimization
        chart = None
        if result and result.best:
            chart = line_chart_svg(
                [(_aware(p.timestamp), p.equity) for p in result.best.equity_curve],
                title="Backtest equity curve, best parameter set",
                value_format=usd_cents,
                reference_lines=[ReferenceLine(result.best.starting_bankroll, "Starting bankroll")],
                chart_id="equity-chart",
            )
        return {
            "status": state.optimizer_status,
            "error": state.optimizer_error,
            "result": result,
            "chart": chart,
            "active_params": await load_active_parameters(engine),
        }

    async def markets_context() -> dict[str, Any]:
        return {
            "markets": state.active_markets(),
            "params": await load_active_parameters(engine),
            "now": datetime.now(timezone.utc),
        }

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        context = {
            "spot_ctx": spot_context(),
            "markets_ctx": await markets_context(),
            "backtest_ctx": await backtest_context(),
        }
        return TEMPLATES.TemplateResponse(request, "index.html", context)

    @app.get("/partials/spot", response_class=HTMLResponse)
    async def spot_partial(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, "partials/spot.html", {"spot_ctx": spot_context()})

    @app.get("/partials/markets", response_class=HTMLResponse)
    async def markets_partial(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, "partials/markets.html", {"markets_ctx": await markets_context()})

    @app.get("/partials/backtest", response_class=HTMLResponse)
    async def backtest_partial(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, "partials/backtest.html", {"backtest_ctx": await backtest_context()})

    @app.post("/optimize", response_class=HTMLResponse)
    async def optimize(request: Request) -> HTMLResponse:
        if state.optimizer_status != "running":
            state.optimizer_status = "running"
            state.optimizer_error = None
            state.optimizer_task = asyncio.create_task(run_optimizer())
        return TEMPLATES.TemplateResponse(request, "partials/backtest.html", {"backtest_ctx": await backtest_context()})

    @app.get("/api/parameters")
    async def active_parameters() -> StrategyParameters:
        return await load_active_parameters(engine)

    return app


app = create_app()
