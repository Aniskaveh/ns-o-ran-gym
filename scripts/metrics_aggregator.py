#!/usr/bin/env python3
"""
Continuously copy per-run SQLite metrics into a single Grafana-friendly database.

Usage:
    python scripts/metrics_aggregator.py \
        --output-root /home/ubadmin/ns-o-ran-gym/output \
        --central-db /home/ubadmin/ns-o-ran-gym/output/grafana_metrics.db
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MetricTable:
    """Metadata that describes how to mirror a source table into the central DB."""

    source: str
    destination: str
    columns: tuple[str, ...]
    state_field: str


METRIC_TABLES: tuple[MetricTable, ...] = (
    MetricTable(
        source="ho_reward_metrics",
        destination="dqn_reward_metrics",
        columns=("timestamp", "step", "reward"),
        state_field="reward_last_step",
    ),
    MetricTable(
        source="ho_training_metrics",
        destination="dqn_training_metrics",
        columns=("timestamp", "step", "loss", "epsilon"),
        state_field="training_last_step",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror per-run metrics databases into a single SQLite file for Grafana."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output",
        help="Root folder that contains ns-O-RAN simulation sub-folders.",
    )
    parser.add_argument(
        "--central-db",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / "grafana_metrics.db",
        help="Path to the consolidated SQLite database Grafana will query.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds to wait between scans. Use 0 to run a single pass and exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Verbosity of the aggregator logs.",
    )
    return parser.parse_args()


def ensure_central_schema(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS dqn_reward_metrics (
            run_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            step INTEGER NOT NULL,
            reward REAL NOT NULL,
            PRIMARY KEY (run_id, step)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS dqn_training_metrics (
            run_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            step INTEGER NOT NULL,
            loss REAL NOT NULL,
            epsilon REAL NOT NULL,
            PRIMARY KEY (run_id, step)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_state (
            run_id TEXT PRIMARY KEY,
            reward_last_step INTEGER NOT NULL DEFAULT 0,
            training_last_step INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_dqn_reward_time ON dqn_reward_metrics(timestamp)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_dqn_training_time ON dqn_training_metrics(timestamp)"
    )
    connection.commit()


def discover_runs(output_root: Path) -> Iterable[tuple[str, Path]]:
    for child in sorted(output_root.iterdir()):
        db_path = child / "database.db"
        if child.is_dir() and db_path.exists():
            yield child.name, db_path


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    cursor = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    )
    return cursor.fetchone() is not None


def ensure_state_row(connection: sqlite3.Connection, run_id: str) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO ingestion_state(run_id) VALUES (?)", (run_id,)
    )
    connection.commit()


def read_last_step(connection: sqlite3.Connection, run_id: str, field: str) -> int:
    cursor = connection.execute(
        f"SELECT {field} FROM ingestion_state WHERE run_id=?", (run_id,)
    )
    row = cursor.fetchone()
    if row is None:
        return 0
    return int(row[0])


def update_last_step(
    connection: sqlite3.Connection, run_id: str, field: str, value: int
) -> None:
    connection.execute(
        f"UPDATE ingestion_state SET {field}=? WHERE run_id=?", (value, run_id)
    )
    connection.commit()


def ingest_metric(
    metric: MetricTable,
    run_id: str,
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
) -> int:
    if not table_exists(source, metric.source):
        logging.debug(
            "Run %s does not expose table %s yet. Skipping.", run_id, metric.source
        )
        return 0

    last_step = read_last_step(destination, run_id, metric.state_field)
    placeholders = ", ".join(metric.columns)
    query = (
        f"SELECT {placeholders} FROM {metric.source} "
        "WHERE step > ? ORDER BY step"
    )
    cursor = source.execute(query, (last_step,))
    rows = cursor.fetchall()
    if not rows:
        return 0

    insert_columns = ", ".join(metric.columns)
    insert_stmt = (
        f"INSERT OR IGNORE INTO {metric.destination} "
        f"(run_id, {insert_columns}) VALUES ({', '.join(['?'] * (len(metric.columns) + 1))})"
    )
    payload = [
        (run_id, *[row[col] for col in metric.columns]) for row in rows
    ]
    destination.executemany(insert_stmt, payload)
    destination.commit()

    new_last_step = max(row["step"] for row in rows)
    update_last_step(destination, run_id, metric.state_field, new_last_step)
    logging.info(
        "Ingested %d %s rows from run %s (up to step %d).",
        len(rows),
        metric.source,
        run_id,
        new_last_step,
    )
    return len(rows)


def process_run(
    run_id: str,
    db_path: Path,
    destination: sqlite3.Connection,
) -> None:
    try:
        with closing(sqlite3.connect(db_path, timeout=5.0)) as source_conn:
            source_conn.row_factory = sqlite3.Row
            ensure_state_row(destination, run_id)
            for metric in METRIC_TABLES:
                try:
                    ingest_metric(metric, run_id, source_conn, destination)
                except sqlite3.OperationalError as exc:
                    logging.warning(
                        "Failed to read %s from run %s (%s). Will retry later.",
                        metric.source,
                        run_id,
                        exc,
                    )
    except sqlite3.OperationalError as exc:
        logging.warning(
            "Cannot open %s (run %s) right now: %s. Skipping this pass.",
            db_path,
            run_id,
            exc,
        )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    output_root = args.output_root.expanduser().resolve()
    central_db = args.central_db.expanduser().resolve()

    if not output_root.exists():
        logging.error("Output root %s does not exist.", output_root)
        return 1

    central_db.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(central_db)) as dest_conn:
        dest_conn.row_factory = sqlite3.Row
        ensure_central_schema(dest_conn)

        while True:
            runs = list(discover_runs(output_root))
            if not runs:
                logging.debug("No ns-O-RAN runs detected under %s.", output_root)
            for run_id, db_path in runs:
                process_run(run_id, db_path, dest_conn)
            if args.poll_interval <= 0:
                break
            time.sleep(args.poll_interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())

