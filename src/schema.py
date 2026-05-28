"""DuckDB schema for intraday 1-min bars.

One table for all instruments (NIFTY spot, futures, options). The
`instrument_kind` discriminator makes per-type queries cheap, and the
composite PK (security_id, ts_minute) is the natural dedup key on
re-pulls.
"""
import duckdb
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "intraday.duckdb"


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def init_db() -> None:
    con = connect()
    con.execute("""
        CREATE TABLE IF NOT EXISTS bars_1min (
          security_id     BIGINT       NOT NULL,
          symbol          VARCHAR      NOT NULL,
          instrument_kind VARCHAR      NOT NULL,
          expiry          DATE,
          strike          DECIMAL(12,2),
          ts_minute       TIMESTAMP    NOT NULL,
          open            DECIMAL(14,4),
          high            DECIMAL(14,4),
          low             DECIMAL(14,4),
          close           DECIMAL(14,4),
          volume          BIGINT,
          oi              BIGINT,
          PRIMARY KEY (security_id, ts_minute)
        );
        CREATE INDEX IF NOT EXISTS idx_bars_ts ON bars_1min(ts_minute);
        CREATE INDEX IF NOT EXISTS idx_bars_sym ON bars_1min(symbol, ts_minute);
    """)
    con.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialised {DB_PATH}")
