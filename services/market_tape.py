"""Aggregate the public Kalshi trade tape into fixed-width bars for storage."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

from db.models import KalshiTapeBar
from execution.kalshi_client import PublicTrade


def _floor(ts: datetime, start: datetime, bar_sec: int) -> datetime:
    return start + timedelta(seconds=int((ts - start).total_seconds()) // bar_sec * bar_sec)


def tape_bars(
    ticker: str, trades: Iterable[PublicTrade], start: datetime, end: datetime, bar_sec: int = 1
) -> list[KalshiTapeBar]:
    """Bars over [start, end) holding count, volume, YES-taker volume and the YES price range/VWAP.
    Seconds without trades are omitted."""
    start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
    buckets: dict[datetime, list[PublicTrade]] = {}
    for t in trades:
        ts = t.created_time.astimezone(timezone.utc)
        if start <= ts < end and t.count > 0:
            buckets.setdefault(_floor(ts, start, bar_sec), []).append(t)
    bars = []
    for bucket_start, group in sorted(buckets.items()):
        volume = sum(t.count for t in group)
        bars.append(
            KalshiTapeBar(
                ticker=ticker,
                start=bucket_start,
                seconds=bar_sec,
                trades=len(group),
                volume=volume,
                taker_yes_volume=sum(t.count for t in group if t.taker_side == "yes"),
                vwap_yes_cents=sum(t.yes_price_cents * t.count for t in group) / volume,
                low_yes_cents=min(t.yes_price_cents for t in group),
                high_yes_cents=max(t.yes_price_cents for t in group),
            )
        )
    return bars
