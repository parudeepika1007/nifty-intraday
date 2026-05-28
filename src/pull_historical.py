"""Pull N years of 1-min NIFTY spot + monthly futures into DuckDB.

Usage:
    python -m src.pull_historical
    python -m src.pull_historical --years 1
"""
from __future__ import annotations

import argparse
import datetime as dt
from datetime import date

import pandas as pd

from .dhan import fetch_intraday
from .instruments import NIFTY_SPOT, Future, list_nifty_futures
from .schema import connect, init_db

DHAN_MAX_DAYS = 90


def _flatten(raw: dict, *, security_id: int, symbol: str, kind: str,
             expiry: date | None, strike: float | None) -> pd.DataFrame:
    if not raw or "timestamp" not in raw or not raw["timestamp"]:
        return pd.DataFrame()
    return pd.DataFrame({
        "security_id": security_id,
        "symbol": symbol,
        "instrument_kind": kind,
        "expiry": expiry,
        "strike": strike,
        "ts_minute": pd.to_datetime(raw["timestamp"], unit="s", utc=True
                                    ).tz_convert("Asia/Kolkata").tz_localize(None),
        "open": raw["open"],
        "high": raw["high"],
        "low": raw["low"],
        "close": raw["close"],
        "volume": raw["volume"],
        "oi": raw.get("open_interest", [None] * len(raw["timestamp"])),
    })


def backfill_instrument(
    con, *, security_id: int, exchange_segment: str, instrument: str,
    symbol: str, kind: str, start: date, end: date,
    expiry: date | None = None, strike: float | None = None,
) -> int:
    total = 0
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + dt.timedelta(days=DHAN_MAX_DAYS - 1), end)
        print(f"  {symbol}  {chunk_start} → {chunk_end}", end="  ")
        raw = fetch_intraday(security_id, exchange_segment, instrument,
                             chunk_start, chunk_end)
        df = _flatten(raw, security_id=security_id, symbol=symbol, kind=kind,
                      expiry=expiry, strike=strike)
        if not df.empty:
            con.register("df_in", df)
            con.execute("INSERT OR REPLACE INTO bars_1min SELECT * FROM df_in")
            con.unregister("df_in")
        print(f"+{len(df)} rows")
        total += len(df)
        chunk_start = chunk_end + dt.timedelta(days=1)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2)
    args = ap.parse_args()

    init_db()
    con = connect()

    end = date.today()
    start = end - dt.timedelta(days=args.years * 365)
    print(f"Backfilling NIFTY 1-min bars  {start} → {end}")

    backfill_instrument(con, **NIFTY_SPOT, kind="spot", start=start, end=end)

    recent = [f for f in list_nifty_futures()
              if f.expiry >= start - dt.timedelta(days=30)][:8]
    for f in recent:
        backfill_instrument(
            con, security_id=f.security_id,
            exchange_segment="NSE_FNO", instrument="FUTIDX",
            symbol=f.symbol, kind="fut", expiry=f.expiry,
            start=max(start, f.expiry - dt.timedelta(days=60)),
            end=min(end, f.expiry),
        )
    con.close()
    print("Done.")


if __name__ == "__main__":
    main()
