"""Index-gap and long-horizon volatility calibration of the Coinbase spot feed."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from core.schemas import BtcSpotTick
from services.btc_spot_feed import BtcSpotFeed
from services.spot_calibration import (
    CalibratedSpot,
    estimate_index_gap,
    gap_observations,
    realized_volatility,
)

T0 = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


def _history(start: datetime, minutes: int, price: float = 82_000.0, step_sec: int = 5) -> list[tuple[datetime, float]]:
    return [(start + timedelta(seconds=i * step_sec), price) for i in range(minutes * 60 // step_sec)]


def test_each_strike_measures_the_gap_over_the_minute_before_its_window_opens() -> None:
    history = _history(T0, 120)
    closes = [T0 + timedelta(minutes=30 + 15 * i) for i in range(5)]
    strikes = [(c, 82_000.0 + 4.0 + i) for i, c in enumerate(closes)]  # index sits $4-8 above Coinbase
    obs = gap_observations(reversed(strikes), history)  # input order doesn't matter
    assert [t for t, _ in obs] == [c - timedelta(minutes=15) for c in closes]
    assert [g for _, g in obs] == pytest.approx([4.0, 5.0, 6.0, 7.0, 8.0])
    assert estimate_index_gap(obs, recent=3) == pytest.approx(7.0)
    assert estimate_index_gap(obs) == pytest.approx(6.0)


def test_gap_needs_price_coverage_and_enough_sane_observations() -> None:
    history = _history(T0, 30)
    late = (T0 + timedelta(hours=3), 82_004.0)  # no Coinbase prices before this window opened
    absurd = (T0 + timedelta(minutes=30), 92_000.0)  # $10k off: a data problem, not a basis
    assert gap_observations([late, absurd], history) == []
    two = gap_observations([(T0 + timedelta(minutes=20), 82_005.0), (T0 + timedelta(minutes=25), 82_006.0)], history)
    assert len(two) == 2 and estimate_index_gap(two) is None


def test_realized_volatility_from_irregular_history_skips_restart_gaps() -> None:
    # Alternating +/-r log returns every 5s: per-second variance r^2 / 5.
    r = 0.0002
    pts, price = [], 82_000.0
    for i in range(721):  # one hour
        pts.append((T0 + timedelta(seconds=5 * i), price))
        price *= math.exp(r if i % 2 == 0 else -r)
    expected = math.sqrt(r * r / 5 * 365 * 24 * 3600)
    now = pts[-1][0]
    assert realized_volatility(pts, now, 3600) == pytest.approx(expected, rel=1e-6)
    # A 10-minute hole (restart) contributes no return, and too little coverage gives None.
    holey = pts[:300] + [(t + timedelta(minutes=10), p * 1.01) for t, p in pts[300:]]
    assert realized_volatility(holey, holey[-1][0], 3600) == pytest.approx(expected, rel=1e-6)
    assert realized_volatility(pts[:100], now, 3600) is None


def test_calibrated_spot_shifts_prices_and_floors_volatility_once_warm() -> None:
    feed = BtcSpotFeed()
    spot = CalibratedSpot(feed)
    assert spot.latest_price() is None and spot.volatility(300) is None
    for i in range(301):
        feed.buffer.add_tick(BtcSpotTick(source="coinbase", price=82_000.0 + (i % 2), timestamp=T0 + timedelta(seconds=i)))
    spot.index_gap = 4.5
    raw_vol = feed.volatility(300)
    assert raw_vol is not None and raw_vol < 0.5
    assert spot.latest_price() == pytest.approx(feed.latest_price() + 4.5)
    assert spot.observed_prices(3) == pytest.approx([p + 4.5 for p in feed.buffer.snapshot(3)])
    assert spot.volatility(300) == pytest.approx(raw_vol)  # no floor yet
    spot.long_volatility = 0.5
    assert spot.volatility(300) == 0.5
    spot.long_volatility = raw_vol / 2
    assert spot.volatility(300) == pytest.approx(raw_vol)  # the floor never lowers volatility
    assert spot.momentum() == feed.momentum() and spot.is_stale(T0 + timedelta(seconds=301)) is False
