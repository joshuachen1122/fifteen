"""Dark-mode HTMX dashboard: live spot vs strike, active 15m options, evaluation results.

Run with: uvicorn web.app:app --reload
The live trader pushes data in through `DashboardState.update_spot` / `update_market`.
"""

import base64
import binascii
import logging
import secrets
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncEngine

from core.schemas import Btc15mMarketState, BtcSpotTick, FairValueOutput, MarketQuote, StrategyParameters
from db.session import init_db, make_engine
from services.agent_stats import AgentActivity, AgentStats
from services.evaluation import EvaluationReport
from services.optimizer import load_active_parameters
from web.charts import ReferenceLine, line_chart_svg, usd

logger = logging.getLogger(__name__)

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
SPOT_HISTORY_LEN = 900


class TradingControls(Protocol):
    """Operator actions exposed on the dashboard (implemented by the live TradingAgent)."""

    def controls_state(self) -> dict[str, Any]: ...
    async def switch_mode(self, live: bool) -> None: ...
    async def set_limits(self, max_trade_usd: float, max_daily_trade_usd: float) -> None: ...
    async def set_agent_enabled(self, enabled: bool) -> None: ...


UNAUTHENTICATED_PATHS = {"/healthz"}  # Fly health checks


def _password_ok(header: Optional[str], password: str) -> bool:
    if not header or not header.lower().startswith("basic "):
        return False
    try:
        _, _, supplied = base64.b64decode(header[6:]).decode().partition(":")
    except (binascii.Error, UnicodeDecodeError):
        return False
    return secrets.compare_digest(supplied.encode(), password.encode())


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
        self.stats: Optional[AgentStats] = None  # refreshed by the live trader
        self.activity: Optional[AgentActivity] = None
        self.evaluation: Optional[EvaluationReport] = None  # latest hourly evaluation
        self.pricing: dict[str, Any] = {}  # which pricing model / market blend the agent is trading on
        self.data_health: dict[str, Any] = {}  # research-data recorder status (refreshed every minute)

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
    controls: Optional[TradingControls] = None,
    password: Optional[str] = None,
) -> FastAPI:
    state = state or DashboardState()
    engine = engine or make_engine()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await init_db(engine)
        yield

    app = FastAPI(title="BTC 15m Kalshi Agent", lifespan=lifespan)
    app.state.dashboard = state
    app.state.engine = engine

    if password:

        @app.middleware("http")
        async def require_password(request: Request, call_next: Any) -> Response:
            if request.url.path in UNAUTHENTICATED_PATHS or _password_ok(
                request.headers.get("authorization"), password
            ):
                return await call_next(request)
            return Response(
                "Authentication required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="btc15m"'}
            )

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
        """Parameter tuning happens in the hourly evaluation; this panel shows its latest result."""
        return {"evaluation": state.evaluation, "active_params": await load_active_parameters(engine)}

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
            "stats": state.stats,
            "activity": state.activity,
            "controls_ctx": controls_context(),
            "evaluation": state.evaluation,
            "pricing": state.pricing,
            "data": state.data_health,
        }
        return TEMPLATES.TemplateResponse(request, "index.html", context)

    def controls_context(error: Optional[str] = None, notice: Optional[str] = None) -> dict[str, Any]:
        if controls is None:
            return {"available": False}
        return {"available": True, "editable": bool(password), "error": error, "notice": notice, **controls.controls_state()}

    def controls_response(request: Request, **kwargs: Any) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, "partials/controls.html", {"controls_ctx": controls_context(**kwargs)})

    async def form(request: Request) -> dict[str, str]:
        return {k: v[0] for k, v in parse_qs((await request.body()).decode()).items()}

    @app.get("/partials/evaluation", response_class=HTMLResponse)
    async def evaluation_partial(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request, "partials/evaluation.html", {"evaluation": state.evaluation, "pricing": state.pricing}
        )

    @app.get("/partials/controls", response_class=HTMLResponse)
    async def controls_partial(request: Request) -> HTMLResponse:
        return controls_response(request)

    @app.post("/controls/mode", response_class=HTMLResponse)
    async def set_mode(request: Request) -> HTMLResponse:
        if controls is None or not password:
            return controls_response(request, error="Set DASHBOARD_PASSWORD to enable dashboard controls.")
        mode = (await form(request)).get("mode")
        if mode not in ("paper", "live"):
            return controls_response(request, error=f"Unknown mode {mode!r}.")
        try:
            await controls.switch_mode(mode == "live")
        except ValueError as exc:
            return controls_response(request, error=str(exc))
        return controls_response(request, notice=f"Switched to {mode.upper()} trading.")

    @app.post("/controls/agent", response_class=HTMLResponse)
    async def set_agent(request: Request) -> HTMLResponse:
        if controls is None or not password:
            return controls_response(request, error="Set DASHBOARD_PASSWORD to enable dashboard controls.")
        action = (await form(request)).get("action")
        if action not in ("stop", "resume"):
            return controls_response(request, error=f"Unknown action {action!r}.")
        try:
            await controls.set_agent_enabled(action == "resume")
        except ValueError as exc:
            return controls_response(request, error=str(exc))
        if action == "stop":
            return controls_response(request, notice="Agent stopped: no new trades and no LLM calls.")
        return controls_response(request, notice="Agent resumed.")

    @app.post("/controls/limits", response_class=HTMLResponse)
    async def set_limits(request: Request) -> HTMLResponse:
        if controls is None or not password:
            return controls_response(request, error="Set DASHBOARD_PASSWORD to enable dashboard controls.")
        data = await form(request)
        try:
            await controls.set_limits(float(data["max_trade_usd"]), float(data["max_daily_trade_usd"]))
        except (KeyError, ValueError) as exc:
            return controls_response(request, error=f"Invalid limits: {exc}")
        return controls_response(request, notice="Spend caps updated.")

    @app.get("/partials/activity", response_class=HTMLResponse)
    async def activity_partial(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, "partials/activity.html", {"activity": state.activity})

    @app.get("/partials/stats", response_class=HTMLResponse)
    async def stats_partial(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, "partials/stats.html", {"stats": state.stats})

    @app.get("/partials/spot", response_class=HTMLResponse)
    async def spot_partial(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, "partials/spot.html", {"spot_ctx": spot_context()})

    @app.get("/partials/markets", response_class=HTMLResponse)
    async def markets_partial(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, "partials/markets.html", {"markets_ctx": await markets_context()})

    @app.get("/partials/data", response_class=HTMLResponse)
    async def data_partial(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, "partials/data.html", {"data": state.data_health})

    @app.get("/partials/backtest", response_class=HTMLResponse)
    async def backtest_partial(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, "partials/backtest.html", {"backtest_ctx": await backtest_context()})

    @app.get("/api/parameters")
    async def active_parameters() -> StrategyParameters:
        return await load_active_parameters(engine)

    return app


app = create_app()
