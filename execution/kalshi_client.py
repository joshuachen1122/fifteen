"""Kalshi Trade API v2 REST client with RSA-PSS auth and a paper-trading mode.

Live order submission only happens when ``trading_enabled=True`` (``TRADING_ENABLED=true``
in the environment). Otherwise orders, cancels, and balance are simulated locally,
while market-data reads still go to the real API so paper trading sees real prices.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self
from urllib.parse import urlparse

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from pydantic import BaseModel, Field

from core.schemas import MarketQuote, OrderRequest
from execution.rate_limiter import DEFAULT_REQUESTS_PER_SECOND, TokenBucketRateLimiter

logger = logging.getLogger(__name__)

PROD_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
PROD_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
DEMO_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"
DEMO_WS_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"

DEFAULT_PAPER_BALANCE_CENTS = 100_000  # $1,000

_STRIKE_SUFFIX = re.compile(r"-[BT](\d+(?:\.\d+)?)$")
_TRUTHY = {"1", "true", "yes", "on"}


class KalshiAPIError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class PaperTradingError(KalshiAPIError):
    """Raised when a simulated order or cancel is rejected."""


def cents_field(raw: dict[str, Any], key: str) -> int | None:
    """Read a price in cents from ``key``, falling back to Kalshi's ``{key}_dollars`` string form."""
    if raw.get(key) is not None:
        return int(raw[key])
    dollars = raw.get(f"{key}_dollars")
    if dollars is None or dollars == "":
        return None
    return round(float(dollars) * 100)


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


class KalshiRequestSigner:
    """Builds Kalshi auth headers: base64(RSA-PSS-SHA256(timestamp_ms + METHOD + path))."""

    def __init__(self, key_id: str, private_key: rsa.RSAPrivateKey) -> None:
        self.key_id = key_id
        self._private_key = private_key

    @classmethod
    def from_pem(cls, key_id: str, pem: bytes | str, password: bytes | None = None) -> KalshiRequestSigner:
        data = pem.encode() if isinstance(pem, str) else pem
        key = serialization.load_pem_private_key(data, password=password)
        if not isinstance(key, rsa.RSAPrivateKey):
            raise ValueError("Kalshi API keys must be RSA private keys")
        return cls(key_id, key)

    @classmethod
    def from_file(cls, key_id: str, path: str | Path, password: bytes | None = None) -> KalshiRequestSigner:
        return cls.from_pem(key_id, Path(path).expanduser().read_bytes(), password)

    def sign(self, message: str) -> str:
        signature = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("ascii")

    def headers(self, method: str, path: str, timestamp_ms: int | None = None) -> dict[str, str]:
        """``path`` is the full URL path (e.g. ``/trade-api/v2/portfolio/balance``); query is ignored."""
        ts = str(timestamp_ms if timestamp_ms is not None else int(time.time() * 1000))
        path_only = path.split("?", 1)[0]
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": self.sign(f"{ts}{method.upper()}{path_only}"),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }


class KalshiMarket(BaseModel):
    ticker: str
    event_ticker: str = ""
    status: str = ""
    strike_price: float
    close_time: datetime
    yes_bid: int | None = None
    yes_ask: int | None = None
    volume: int = 0
    result: str = ""  # "yes" | "no" once determined, else ""
    settlement_price: float | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> KalshiMarket:
        ticker = raw["ticker"]
        strike = raw.get("floor_strike")
        if strike is None:
            match = _STRIKE_SUFFIX.search(ticker)
            if match is None:
                raise ValueError(f"cannot determine strike for market {ticker}")
            strike = float(match.group(1))
        return cls(
            ticker=ticker,
            event_ticker=raw.get("event_ticker", ""),
            status=raw.get("status", ""),
            strike_price=float(strike),
            close_time=raw["close_time"],
            yes_bid=cents_field(raw, "yes_bid"),
            yes_ask=cents_field(raw, "yes_ask"),
            volume=int(float(raw.get("volume", raw.get("volume_fp", 0)) or 0)),
            result=raw.get("result") or "",
            settlement_price=_float_or_none(raw.get("expiration_value")),
        )

    def to_quote(self, updated_at: datetime | None = None) -> MarketQuote | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return MarketQuote(
            ticker=self.ticker,
            strike_price=self.strike_price,
            yes_bid=self.yes_bid,
            yes_ask=self.yes_ask,
            volume=self.volume,
            updated_at=updated_at or datetime.now(timezone.utc),
        )


class OrderResult(BaseModel):
    order_id: str
    client_order_id: str | None = None
    ticker: str
    side: Literal["yes", "no"]
    status: str
    count: int
    price_cents: int
    filled_count: int = 0
    paper: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> OrderResult:
        side = raw["side"]
        price = cents_field(raw, f"{side}_price") or 0
        initial = raw.get("initial_count", raw.get("count"))
        remaining = raw.get("remaining_count", 0)
        filled = raw.get("fill_count")
        if filled is None and initial is not None:
            filled = int(initial) - int(remaining or 0)
        return cls(
            order_id=raw["order_id"],
            client_order_id=raw.get("client_order_id"),
            ticker=raw["ticker"],
            side=side,
            status=raw.get("status", "unknown"),
            count=int(initial if initial is not None else remaining or 0),
            price_cents=price,
            filled_count=int(filled or 0),
            raw=raw,
        )


def limit_price_cents(order: OrderRequest) -> int:
    """Per-contract limit price implied by ``max_cost_cents / count`` (rounded down)."""
    if order.side not in ("yes", "no"):
        raise ValueError(f"order side must be 'yes' or 'no', got {order.side!r}")
    if order.count < 1:
        raise ValueError(f"order count must be >= 1, got {order.count}")
    price = order.max_cost_cents // order.count
    if not 1 <= price <= 99:
        raise ValueError(
            f"implied limit price {price}c (max_cost_cents={order.max_cost_cents}, count={order.count}) "
            "is outside Kalshi's 1-99c range"
        )
    return price


class KalshiClient:
    def __init__(
        self,
        *,
        base_url: str = DEMO_BASE_URL,
        ws_url: str = DEMO_WS_URL,
        signer: KalshiRequestSigner | None = None,
        trading_enabled: bool = False,
        rate_limiter: TokenBucketRateLimiter | None = None,
        http_client: httpx.AsyncClient | None = None,
        paper_starting_balance_cents: int = DEFAULT_PAPER_BALANCE_CENTS,
        timeout: float = 10.0,
    ) -> None:
        if trading_enabled and signer is None:
            raise ValueError("live trading requires Kalshi API credentials (signer)")
        self.base_url = base_url.rstrip("/")
        self.ws_url = ws_url
        self.trading_enabled = trading_enabled
        self._signer = signer
        self._base_path = urlparse(self.base_url).path
        self._rate_limiter = rate_limiter or TokenBucketRateLimiter(DEFAULT_REQUESTS_PER_SECOND)
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout)
        self._paper_balance_cents = paper_starting_balance_cents
        self._paper_orders: dict[str, OrderResult] = {}

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, **overrides: Any) -> KalshiClient:
        """Configure from ``TRADING_ENABLED``, ``KALSHI_ENV`` (demo|prod), ``KALSHI_API_KEY_ID``,
        (or ``KALSHI_KEY_ID``), ``KALSHI_PRIVATE_KEY_PATH`` or ``KALSHI_PRIVATE_KEY`` (PEM), ``KALSHI_RATE_LIMIT_RPS``,
        and ``PAPER_STARTING_BALANCE_CENTS``."""
        env = dict(os.environ) if env is None else env
        trading_enabled = env.get("TRADING_ENABLED", "false").strip().lower() in _TRUTHY
        kalshi_env = env.get("KALSHI_ENV", "demo").strip().lower()
        if kalshi_env not in ("demo", "prod"):
            raise ValueError(f"KALSHI_ENV must be 'demo' or 'prod', got {kalshi_env!r}")

        signer: KalshiRequestSigner | None = None
        key_id = env.get("KALSHI_API_KEY_ID") or env.get("KALSHI_KEY_ID")
        if key_id and env.get("KALSHI_PRIVATE_KEY_PATH"):
            signer = KalshiRequestSigner.from_file(key_id, env["KALSHI_PRIVATE_KEY_PATH"])
        elif key_id and env.get("KALSHI_PRIVATE_KEY"):
            signer = KalshiRequestSigner.from_pem(key_id, env["KALSHI_PRIVATE_KEY"])

        kwargs: dict[str, Any] = {
            "base_url": PROD_BASE_URL if kalshi_env == "prod" else DEMO_BASE_URL,
            "ws_url": PROD_WS_URL if kalshi_env == "prod" else DEMO_WS_URL,
            "signer": signer,
            "trading_enabled": trading_enabled,
            "rate_limiter": TokenBucketRateLimiter(
                float(env.get("KALSHI_RATE_LIMIT_RPS", DEFAULT_REQUESTS_PER_SECOND))
            ),
            "paper_starting_balance_cents": int(
                env.get("PAPER_STARTING_BALANCE_CENTS", DEFAULT_PAPER_BALANCE_CENTS)
            ),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def is_paper(self) -> bool:
        return not self.trading_enabled

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    def ws_auth_headers(self) -> dict[str, str]:
        if self._signer is None:
            return {}
        return self._signer.headers("GET", urlparse(self.ws_url).path)

    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        auth: bool = True,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if auth:
            if self._signer is None:
                raise KalshiAPIError(f"{method} {endpoint} requires Kalshi API credentials")
            headers.update(self._signer.headers(method, self._base_path + endpoint))

        await self._rate_limiter.acquire()
        try:
            response = await self._http.request(
                method, self.base_url + endpoint, params=params, json=json, headers=headers
            )
        except httpx.HTTPError as exc:
            raise KalshiAPIError(f"{method} {endpoint} failed: {exc}") from exc

        try:
            body: Any = response.json() if response.content else {}
        except ValueError:
            body = response.text
        if response.is_error:
            raise KalshiAPIError(
                f"{method} {endpoint} returned HTTP {response.status_code}: {body}",
                status_code=response.status_code,
                body=body,
            )
        return body if isinstance(body, dict) else {"data": body}

    async def get_balance(self) -> int:
        """Available balance in cents."""
        if self.is_paper:
            return self._paper_balance_cents
        body = await self._request("GET", "/portfolio/balance")
        return int(body["balance"])

    async def get_markets(
        self,
        series_ticker: str = "KXBTC15M",
        status: str = "open",
        *,
        page_limit: int = 200,
    ) -> list[KalshiMarket]:
        markets: list[KalshiMarket] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"series_ticker": series_ticker, "status": status, "limit": page_limit}
            if cursor:
                params["cursor"] = cursor
            body = await self._request("GET", "/markets", params=params, auth=self._signer is not None)
            for raw in body.get("markets", []):
                try:
                    markets.append(KalshiMarket.from_api(raw))
                except (KeyError, ValueError) as exc:
                    logger.warning("skipping unparseable market %s: %s", raw.get("ticker"), exc)
            cursor = body.get("cursor") or None
            if cursor is None:
                return markets

    async def get_market(self, ticker: str) -> KalshiMarket:
        body = await self._request("GET", f"/markets/{ticker}", auth=self._signer is not None)
        return KalshiMarket.from_api(body["market"])

    def credit_paper_balance(self, cents: int) -> None:
        """Return settlement payouts to the paper balance (live balances are credited by Kalshi)."""
        if self.is_paper:
            self._paper_balance_cents += cents

    async def submit_order(self, order: OrderRequest, *, client_order_id: str | None = None) -> OrderResult:
        """Submit a buy limit order priced at ``max_cost_cents // count`` per contract."""
        price = limit_price_cents(order)
        client_order_id = client_order_id or str(uuid.uuid4())
        if self.is_paper:
            return self._paper_submit(order, price, client_order_id)

        payload: dict[str, Any] = {
            "ticker": order.ticker,
            "client_order_id": client_order_id,
            "action": "buy",
            "side": order.side,
            "count": order.count,
            "type": "limit",
            f"{order.side}_price": price,
        }
        logger.info("submitting live order %s", payload)
        body = await self._request("POST", "/portfolio/orders", json=payload)
        return OrderResult.from_api(body["order"])

    async def cancel_order(self, order_id: str) -> OrderResult:
        if self.is_paper:
            return self._paper_cancel(order_id)
        body = await self._request("DELETE", f"/portfolio/orders/{order_id}")
        return OrderResult.from_api(body["order"])

    def _paper_submit(self, order: OrderRequest, price: int, client_order_id: str) -> OrderResult:
        cost = price * order.count
        if cost > self._paper_balance_cents:
            raise PaperTradingError(
                f"insufficient paper balance: order costs {cost}c, balance is {self._paper_balance_cents}c",
                status_code=400,
            )
        # Paper fills are immediate and complete at the limit price.
        self._paper_balance_cents -= cost
        result = OrderResult(
            order_id=f"paper-{uuid.uuid4()}",
            client_order_id=client_order_id,
            ticker=order.ticker,
            side=order.side,  # type: ignore[arg-type]  # validated by limit_price_cents
            status="executed",
            count=order.count,
            price_cents=price,
            filled_count=order.count,
            paper=True,
        )
        self._paper_orders[result.order_id] = result
        logger.info("paper order filled: %s %dx %s @ %dc", order.side, order.count, order.ticker, price)
        return result

    def _paper_cancel(self, order_id: str) -> OrderResult:
        existing = self._paper_orders.get(order_id)
        if existing is None:
            raise PaperTradingError(f"paper order {order_id} not found", status_code=404)
        if existing.status != "resting":
            raise PaperTradingError(f"paper order {order_id} is {existing.status}; cannot cancel", status_code=400)
        canceled = existing.model_copy(update={"status": "canceled"})
        self._paper_orders[order_id] = canceled
        return canceled
