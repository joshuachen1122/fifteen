"""Async SQLite engine helpers."""

import os

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel

import db.models  # noqa: F401  (registers tables on SQLModel.metadata)

DEFAULT_DB_URL = os.environ.get("BTC15M_DB_URL", "sqlite+aiosqlite:///./btc15m.db")


def make_engine(url: str = DEFAULT_DB_URL) -> AsyncEngine:
    return create_async_engine(url)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
