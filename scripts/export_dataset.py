"""Export the research data for retraining.

    python -m scripts.export_dataset --pull            # copy the production DB, then export
    python -m scripts.export_dataset --db agent.db     # export a local copy

Writes gzipped CSVs to data/<timestamp>/: `training` (one labelled row per snapshot, see
services/dataset.py) plus the raw tables. The production copy is taken with SQLite's online
backup on the Fly machine, so it is consistent while the agent keeps writing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from services.dataset import export

APP = "btc-15m-kalshi-agent"
REMOTE_COPY = "/tmp/research-copy.db"
BACKUP = (
    "import sqlite3; s=sqlite3.connect('/data/agent.db'); d=sqlite3.connect('{dst}'); s.backup(d); d.close(); print('ok')"
)


def pull(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)  # `fly ssh sftp get` refuses to overwrite
    subprocess.run(
        ["fly", "ssh", "console", "-a", APP, "-C", f'python3 -c "{BACKUP.format(dst=REMOTE_COPY)}"'], check=True
    )
    subprocess.run(["fly", "ssh", "sftp", "get", REMOTE_COPY, str(dest), "-a", APP], check=True)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--db", type=Path, help="local SQLite file")
    source.add_argument("--pull", action="store_true", help="copy the production DB from Fly first")
    parser.add_argument("--out", type=Path, help="output directory (default data/<UTC timestamp>)")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or Path("data") / stamp
    db = pull(out / "agent.db") if args.pull else args.db
    counts = export(db, out)
    for name, n in counts.items():
        print(f"{name:>14}: {n:,} rows")
    print(f"written to {out}/", file=sys.stderr)


if __name__ == "__main__":
    main()
