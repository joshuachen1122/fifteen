"""Async token-bucket rate limiter for outbound Kalshi API calls."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Self

DEFAULT_REQUESTS_PER_SECOND = 10.0

# Absorbs float drift in refill math; without it a bucket at 0.9999999 tokens can spin on
# sleeps too small to advance the clock.
_EPSILON = 1e-9


class TokenBucketRateLimiter:
    """Token bucket that refills continuously at ``rate`` tokens per second.

    ``capacity`` bounds the burst size (defaults to one second's worth of tokens).
    Waiters are served FIFO: the internal lock is held while sleeping, so a later
    caller can never starve an earlier one.
    """

    def __init__(
        self,
        rate: float = DEFAULT_REQUESTS_PER_SECOND,
        capacity: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        resolved_capacity = rate if capacity is None else capacity
        if resolved_capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {resolved_capacity}")

        self._rate = float(rate)
        self._capacity = float(resolved_capacity)
        self._tokens = self._capacity
        self._clock = clock
        self._sleep = sleep
        self._last_refill = clock()
        self._lock = asyncio.Lock()

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def capacity(self) -> float:
        return self._capacity

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self._tokens

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now

    def _validate(self, tokens: float) -> None:
        if tokens <= 0:
            raise ValueError(f"tokens must be positive, got {tokens}")
        if tokens > self._capacity:
            raise ValueError(f"cannot acquire {tokens} tokens; bucket capacity is {self._capacity}")

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Take ``tokens`` immediately if available; never waits."""
        self._validate(tokens)
        if self._lock.locked():
            return False
        self._refill()
        if self._tokens + _EPSILON >= tokens:
            self._tokens = max(0.0, self._tokens - tokens)
            return True
        return False

    async def acquire(self, tokens: float = 1.0) -> None:
        """Wait until ``tokens`` are available, then consume them."""
        self._validate(tokens)
        async with self._lock:
            while True:
                self._refill()
                if self._tokens + _EPSILON >= tokens:
                    self._tokens = max(0.0, self._tokens - tokens)
                    return
                await self._sleep((tokens - self._tokens) / self._rate)

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None
