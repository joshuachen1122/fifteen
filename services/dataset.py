"""Turn the recorded research data into a training table (offline; reads a SQLite copy of the DB).

One row per market snapshot of a settled contract, labelled with the outcome, joined with the
index-exchange sample at (or just before) that second and the Kalshi tape for that second:

    label            1 if the contract settled YES
    yes_bid/ask      cents; mid_prob = mid / 100
    spot_price       calibrated Coinbase (what the model saw); coinbase_raw = spot - index_gap
    composite_spot   median mid across the index exchanges; per-exchange mids as columns
    observed_*       the part of the settlement average already printed (final minute)
    tape_*           Kalshi prints in that second

Use `python -m scripts.export_dataset` to write it out, or `build_training_frame` directly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from services.exchange_recorder import SOURCES

INDEX_JOIN_TOLERANCE = pd.Timedelta(seconds=2)


def _read(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


def _utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, format="ISO8601")


def load_tables(db_path: str | Path) -> dict[str, pd.DataFrame]:
    """The raw research tables (timestamps parsed as UTC)."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        names = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
        tables = {
            "snapshots": _read(conn, "select * from market_snapshots"),
            "outcomes": _read(conn, "select * from contract_outcomes"),
            "index_samples": _read(conn, "select * from index_samples") if "index_samples" in names else pd.DataFrame(),
            "tape_bars": _read(conn, "select * from kalshi_tape_bars") if "kalshi_tape_bars" in names else pd.DataFrame(),
            "spot": _read(conn, "select source, price, timestamp from btc_price_history"),
        }
    for name, cols in {
        "snapshots": ("ts", "close_time"),
        "outcomes": ("close_time", "recorded_at"),
        "index_samples": ("ts",),
        "tape_bars": ("start",),
        "spot": ("timestamp",),
    }.items():
        for c in cols:
            if c in tables[name]:
                tables[name][c] = _utc(tables[name][c])
    return tables


def build_training_frame(db_path: str | Path) -> pd.DataFrame:
    t = load_tables(db_path)
    snaps, outcomes = t["snapshots"], t["outcomes"]
    df = snaps.merge(
        outcomes[["ticker", "settled_yes", "settlement_price"]], on="ticker", how="inner", validate="many_to_one"
    )
    df["label"] = df.pop("settled_yes").astype(int)
    df["mid_prob"] = (df["yes_bid"] + df["yes_ask"]) / 200
    df["spread_cents"] = df["yes_ask"] - df["yes_bid"]
    df["coinbase_raw"] = df["spot_price"] - df["index_gap"].fillna(0.0)
    df["spot_minus_strike"] = df["spot_price"] - df["strike_price"]
    df = df.sort_values("ts").reset_index(drop=True)

    index = t["index_samples"]
    if not index.empty:
        cols = ["ts", *[c for c in SOURCES if c in index], "composite", "sources"]
        joined = pd.merge_asof(
            df[["ts"]].reset_index(),
            index[cols].sort_values("ts").rename(columns={"composite": "index_composite", "sources": "index_sources"}),
            on="ts",
            direction="backward",
            tolerance=INDEX_JOIN_TOLERANCE,
        ).set_index("index")
        df = df.join(joined.drop(columns="ts"))
        df["composite_minus_strike"] = df["index_composite"] - df["strike_price"]

    tape = t["tape_bars"]
    if not tape.empty:
        tape = tape.assign(second=tape["start"].dt.floor("s"))[
            ["ticker", "second", "trades", "volume", "taker_yes_volume", "vwap_yes_cents"]
        ].rename(columns=lambda c: c if c in ("ticker", "second") else f"tape_{c}")
        df = df.assign(second=df["ts"].dt.floor("s")).merge(tape, on=["ticker", "second"], how="left")
        df = df.drop(columns="second")
    return df


def export(db_path: str | Path, out_dir: str | Path) -> dict[str, int]:
    """Write the training frame and the raw research tables as gzipped CSVs. Returns row counts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = {"training": build_training_frame(db_path), **load_tables(db_path)}
    for name, frame in frames.items():
        frame.to_csv(out / f"{name}.csv.gz", index=False)
    return {name: len(frame) for name, frame in frames.items()}
