"""Async SQLite engine and session management."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from core.constants import DEFAULT_DATABASE_URL

# Import models so their tables register on SQLModel.metadata.
from db import models as _models  # noqa: F401


def _is_memory_url(url: str) -> bool:
    return ":memory:" in url or url.rstrip("/").endswith("sqlite+aiosqlite:")


class Database:
    def __init__(self, url: str | None = None, *, echo: bool = False) -> None:
        self.url = url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
        kwargs: dict[str, Any] = {"echo": echo}
        if self.url.startswith("sqlite") and _is_memory_url(self.url):
            # One shared connection so every session sees the same in-memory DB.
            kwargs.update(poolclass=StaticPool, connect_args={"check_same_thread": False})
        self.engine: AsyncEngine = create_async_engine(self.url, **kwargs)
        if self.url.startswith("sqlite"):
            event.listen(self.engine.sync_engine, "connect", _sqlite_pragmas)
        self._sessionmaker = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._sessionmaker() as session:
            yield session

    async def dispose(self) -> None:
        await self.engine.dispose()


def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    # WAL lets the dashboard read while the trading loop writes; no-op for :memory:.
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


_default_db: Database | None = None


def get_database() -> Database:
    global _default_db
    if _default_db is None:
        _default_db = Database()
    return _default_db


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI-style dependency yielding a session on the default database."""
    async with get_database().session() as session:
        yield session
