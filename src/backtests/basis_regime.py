"""Backtest: NIFTY intraday basis regime.

Hypothesis: basis = (NIFTY_FUT - NIFTY_spot) carries directional info
at the 1-min level. Extreme basis z-scores predict short-term forward
spot direction.

Reports: 5-row quintile table with n, hit%, median bps, mean bps.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from ..schema import DB_PATH


def load_aligned(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = """
    WITH front AS (
        SELECT ts_minute, close, expiry, symbol,
               ROW_NUMBER() OVER (PARTITION BY ts_minute ORDER BY expiry ASC) AS rn
        FROM bars_1min
        WHERE instrument_kind = 'fut' AND expiry >= ts_minute::DATE
    )
    SELECT s.ts_minute, s.close AS spot, f.close AS fut, f.symbol AS fut_sym
    FROM bars_1min s
    JOIN front f ON f.ts_minute = s.ts_minute AND f.rn = 1
    WHERE s.symbol = 'NIFTY_SPOT'
    ORDER BY s.ts_minute
    """
    return con.sql(sql).df()


def compute_basis_signals(df: pd.DataFrame, zwin: int = 60) -> pd.DataFrame:
    df = df.copy()
    df["basis"] = df["fut"] - df["spot"]
    df["basis_pct"] = df["basis"] / df["spot"] * 100.0
    df["trade_date"] = df["ts_minute"].dt.date

    def zscore_intraday(g: pd.DataFrame) -> pd.DataFrame:
        b = g["basis"]
        roll_mean = b.rolling(zwin, min_periods=zwin // 2).mean()
        roll_std = b.rolling(zwin, min_periods=zwin // 2).std()
        g["basis_z"] = (b - roll_mean) / roll_std
        return g

    df = df.groupby("trade_date", group_keys=False).apply(zscore_intraday)
    df["fwd_15m_bps"] = (df["spot"].shift(-15) / df["spot"] - 1) * 10000
    df["fwd_30m_bps"] = (df["spot"].shift(-30) / df["spot"] - 1) * 10000
    return df


def report(df: pd.DataFrame) -> None:
    df = df.dropna(subset=["basis_z", "fwd_30m_bps"]).copy()
    df["z_bin"] = pd.qcut(df["basis_z"], q=5,
                          labels=["Q1_low", "Q2", "Q3_mid", "Q4", "Q5_high"])

    rows = []
    for label, grp in df.groupby("z_bin", observed=True):
        rows.append({
            "z_bin": label,
            "n": len(grp),
            "z_median": grp["basis_z"].median(),
            "hit_15m": (grp["fwd_15m_bps"] > 0).mean() * 100,
            "mean_15m_bps": grp["fwd_15m_bps"].mean(),
            "hit_30m": (grp["fwd_30m_bps"] > 0).mean() * 100,
            "mean_30m_bps": grp["fwd_30m_bps"].mean(),
        })
    out = pd.DataFrame(rows)
    print("\nBasis Regime backtest — forward returns by basis z-quintile")
    print("=" * 78)
    print(out.to_string(index=False, float_format="%.2f"))

    top = out.iloc[-1]
    bot = out.iloc[0]
    print(f"\nQ5 - Q1 spread:  15m {top['mean_15m_bps'] - bot['mean_15m_bps']:+.2f} bps  "
          f"30m {top['mean_30m_bps'] - bot['mean_30m_bps']:+.2f} bps")
    print(f"Q5 hit-rate uplift vs Q1:  15m {top['hit_15m'] - bot['hit_15m']:+.1f}pp  "
          f"30m {top['hit_30m'] - bot['hit_30m']:+.1f}pp")


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    raw = load_aligned(con)
    print(f"Loaded {len(raw):,} aligned 1-min bars  "
          f"({raw['ts_minute'].min()} → {raw['ts_minute'].max()})")
    df = compute_basis_signals(raw)
    report(df)
    con.close()


if __name__ == "__main__":
    main()
