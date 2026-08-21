import sqlite3
from datetime import datetime
from pathlib import Path

from taurus.data.schemas import BarInterval, PriceBar


class SQLitePriceBarRepository:
    # SQLite-backed storage for Taurus price bars.

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS price_bars (
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    source TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, timestamp, source, interval)
                )
                """
            )

    def save_bars(self, bars: list[PriceBar]) -> None:
        ingested_at = datetime.now().astimezone().isoformat()

        rows = [
            (
                bar.symbol,
                bar.timestamp.isoformat(),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.source,
                bar.interval.value,
                ingested_at,
            )
            for bar in bars
        ]

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO price_bars (
                    symbol,
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    source,
                    interval,
                    ingested_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def get_bars(
            self,
            symbol: str,
            interval: BarInterval,
        ) -> list[PriceBar]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    symbol,
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    source,
                    interval
                FROM price_bars
                WHERE symbol = ? AND interval = ?
                ORDER BY timestamp ASC
                """,
                (symbol, interval.value),
            ).fetchall()

        return [
            PriceBar(
                symbol=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                open=row[2],
                high=row[3],
                low=row[4],
                close=row[5],
                volume=row[6],
                source=row[7],
                interval=BarInterval(row[8]),
            )
            for row in rows
        ]