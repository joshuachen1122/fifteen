"""Replay the final-minute strategy over recorded snapshots under alternative rules.

    python -m scripts.backtest_final_minute --db data/<stamp>/agent.db

For every settled contract with recorder data, walks its snapshots between 60 and 30 seconds
before the close and enters on the first one where a side's settlement-model edge clears
`min_gross_edge` (one entry per contract, like the agent). Variants change the spot input
(Coinbase + strike-based gap, as traded, vs the six-exchange composite) and the minimum price
paid for the side bought. P&L is net of the taker fee, sized with quarter-Kelly against a fixed
bankroll so variants are comparable. The "coinbase, no floor" row should reproduce the paper
trades the agent actually placed (see `matched`).

On the first 2.5 days the composite tracked the settlement average better over whole windows but
priced the final minute worse (Brier 0.077-0.091 vs 0.055): the non-Coinbase quotes it takes the
median of changed in only 8-16% of seconds, so it lagged the moves that decide close contracts.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from core.schemas import Btc15mMarketState, MarketQuote, StrategyParameters
from pricing.settlement import SETTLEMENT_WINDOW_SEC, price_settlement_average
from services.risk_engine import KALSHI_TAKER_FEE_RATE, kelly_fraction

ENTRY_MAX_SEC, ENTRY_MIN_SEC = 60, 30
MIN_INDEX_SOURCES = 3


@dataclass(frozen=True)
class Variant:
    name: str
    spot: str  # "coinbase" | "composite"
    min_price: int = 1


def load(db: str, since: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        snaps = pd.read_sql(
            "select s.*, o.settled_yes from market_snapshots s join contract_outcomes o using(ticker) "
            "where s.close_time >= ? and s.seconds_remaining between ? and ? order by s.ts",
            con, params=(since, ENTRY_MIN_SEC, ENTRY_MAX_SEC),
        )
        index = pd.read_sql("select ts, composite, sources from index_samples where ts >= ?", con, params=(since,))
        trades = pd.read_sql(
            "select ticker, side, avg_fill_price_cents, filled_count, realized_pnl_cents from trades "
            "where filled_count > 0 and is_paper and expiration_time >= ?", con, params=(since,),
        )
    for df, cols in ((snaps, ("ts", "close_time")), (index, ("ts",))):
        for c in cols:
            df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    return snaps, index, trades


def composite_inputs(snaps: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Per snapshot: the composite at that second and its mean over the elapsed averaging window."""
    index = index[index.sources >= MIN_INDEX_SOURCES].sort_values("ts")
    ts = index.ts.astype("int64").to_numpy()  # ns since epoch
    comp = index.composite.to_numpy()
    csum = np.concatenate([[0.0], np.cumsum(comp)])
    snap_ts = snaps.ts.astype("int64").to_numpy()
    window_start = (snaps.close_time - timedelta(seconds=SETTLEMENT_WINDOW_SEC)).astype("int64").to_numpy()
    j = np.searchsorted(ts, snap_ts, side="right")
    start = np.searchsorted(ts, window_start)
    fresh = (j > 0) & (snap_ts - ts[np.maximum(j - 1, 0)] <= 2 * 10**9)
    spot = np.where(fresh, comp[np.maximum(j - 1, 0)], np.nan)
    n = j - start
    obs = np.where(n > 0, (csum[j] - csum[np.minimum(start, j)]) / np.maximum(n, 1), np.nan)
    return snaps.assign(composite_now=spot, composite_observed=obs)


def p_yes(r: pd.Series, spot: float, observed: Optional[float], params: StrategyParameters) -> float:
    state = Btc15mMarketState(
        ticker=r.ticker, strike_price=r.strike_price, expiration_time=r.close_time,
        seconds_remaining=int(r.seconds_remaining), current_spot_price=spot,
        historical_volatility=r.historical_volatility, momentum_1m=r.momentum_1m,
        order_book_imbalance=r.order_book_imbalance,
    )
    quote = MarketQuote(ticker=r.ticker, strike_price=r.strike_price, yes_bid=int(r.yes_bid),
                        yes_ask=int(r.yes_ask), volume=0, updated_at=r.ts)
    observed_prices = [] if observed is None or np.isnan(observed) else [observed]
    return price_settlement_average(state, quote, params, observed=observed_prices).model_probability


def simulate(snaps: pd.DataFrame, v: Variant, params: StrategyParameters, bankroll_cents: int) -> pd.DataFrame:
    rows = []
    for ticker, g in snaps.groupby("ticker", sort=False):
        for r in g.itertuples():
            if v.spot == "composite":
                if np.isnan(r.composite_now):
                    continue
                p = p_yes(r, r.composite_now, r.composite_observed if r.seconds_remaining < SETTLEMENT_WINDOW_SEC else None, params)
            else:
                p = r.settlement_model_probability  # what the agent priced at the time
                if p is None or np.isnan(p):
                    continue
            sides = [("yes", int(r.yes_ask), p), ("no", 100 - int(r.yes_bid), 1 - p)]
            pick = [(s, a, q) for s, a, q in sides if q - a / 100 > params.min_gross_edge and v.min_price <= a <= 99]
            if not pick:
                continue
            side, ask, q = max(pick, key=lambda x: x[2] - x[1] / 100)
            fraction = min(kelly_fraction(q, ask) * params.kelly_scaling, params.max_trade_bankroll_pct)
            fee = KALSHI_TAKER_FEE_RATE * ask * (100 - ask) / 100  # cents per contract
            count = int(fraction * bankroll_cents // (ask + fee))
            if count <= 0:
                continue
            won = bool(r.settled_yes) == (side == "yes")
            per_contract = (100 if won else 0) - ask - fee
            rows.append(dict(ticker=ticker, side=side, ask=ask, p_win=q, won=won, count=count,
                             per_contract=per_contract, pnl=per_contract * count, secs=r.seconds_remaining))
            break
    return pd.DataFrame(rows)


def summarize(name: str, t: pd.DataFrame, actual: pd.DataFrame, rng: np.random.Generator) -> dict:
    if t.empty:
        return dict(variant=name, trades=0)
    boot = [t.per_contract.sample(len(t), replace=True, random_state=rng.integers(1 << 31)).mean() for _ in range(2000)]
    keys = set(zip(actual.ticker, actual.side))
    return dict(
        variant=name, trades=len(t), win_rate=t.won.mean(), avg_price=t.ask.mean(),
        model_win=t.p_win.mean(), per_contract_c=t.per_contract.mean(),
        ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
        kelly_pnl_usd=t.pnl.sum() / 100, matched=sum((a, b) in keys for a, b in zip(t.ticker, t.side)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--since", default="2026-10-10 18:00", help="first contract close with recorder data (UTC)")
    ap.add_argument("--bankroll", type=float, default=700.0, help="fixed bankroll for Kelly sizing, USD")
    ap.add_argument("--floors", default="1,10,15,20,25,30")
    args = ap.parse_args()

    params = StrategyParameters()
    snaps, index, actual = load(args.db, args.since)
    snaps = composite_inputs(snaps, index)
    floors = [int(x) for x in args.floors.split(",")]
    variants = [Variant(f"{s}, floor {f}c", s, f) for s in ("coinbase", "composite") for f in floors]
    rng = np.random.default_rng(0)
    results = {v.name: simulate(snaps, v, params, int(args.bankroll * 100)) for v in variants}
    table = pd.DataFrame([summarize(n, t, actual, rng) for n, t in results.items()])

    pd.set_option("display.width", 200)
    print(f"contracts with final-minute snapshots: {snaps.ticker.nunique()}; actual paper trades: {len(actual)} "
          f"(${actual.realized_pnl_cents.sum() / 100:.2f})\n")
    print(table.round(3).to_string(index=False))
    for name in (variants[0].name, f"composite, floor {floors[0]}c"):
        t = results[name]
        if not t.empty:
            t = t.assign(bucket=pd.cut(t.ask, [0, 10, 20, 40, 60, 80, 100]))
            print(f"\n{name} by price paid:")
            print(t.groupby("bucket", observed=True).agg(
                n=("won", "size"), won=("won", "mean"), model=("p_win", "mean"), per_contract=("per_contract", "mean")
            ).round(3).to_string())


if __name__ == "__main__":
    main()
