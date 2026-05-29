"""Backtest: NIFTY Opening Range Breakout (idea #1), spot-only variant.

Market logic (CLAUDE.md §4 #1). The first 15 min (09:15-09:30 IST) absorbs
overnight news; by 09:30 the session's directional bias is visible. A break
of that opening range has positive expectancy when the day-type cooperates.

This is the SPOT-ONLY cut of the idea. The full design also wanted:
  * a basis-premium confirmation  -> deferred: no intraday futures history
                                      (Dhan live master omits expired contracts)
  * a volume > 1.2x median filter  -> dropped: NIFTY index 1-min `volume` is
                                      0 on ~72% of bars, unusable as a filter
Both hooks are noted here so a future session can add them when the data lands.

Method (walk-forward, no peeking):
  * Opening range = high/low of the first `or_minutes` bars from 09:15.
  * First bar AFTER the range whose high pierces OR-high (long) or low pierces
    OR-low (short) is the breakout. Entry = the pierced level (classic ORB).
  * Forward return is SIGNED by breakout direction: positive = the break
    continued, negative = it faded. Measured at +`fwd_min` and at EOD close.
  * Everything resets per session. Inside days (no break) produce no signal.

§4 flags two failure conditions we test head-on by bucketing:
  * Range days (narrow OR)  -> the OR-width quintile table.
  * Gap days (>~0.6%)       -> the gap bucket table.

Reports quintile/bucket tables with n, hit%, median bps, mean bps — same
shape as basis_regime.py and the EOD repo's summarize().
"""
from __future__ import annotations

from datetime import time

import duckdb
import numpy as np
import pandas as pd

from ..schema import DB_PATH

SESSION_OPEN = time(9, 15)
OR_MINUTES = 15        # opening-range window length (minutes)
FWD_MIN = 30           # forward horizon after entry (minutes)


def load_spot(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = """
    SELECT ts_minute, open, high, low, close
    FROM bars_1min
    WHERE symbol = 'NIFTY_SPOT'
    ORDER BY ts_minute
    """
    df = con.sql(sql).df()
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df


def compute_orb_signals(df: pd.DataFrame, *, or_minutes: int = OR_MINUTES,
                        fwd_min: int = FWD_MIN) -> pd.DataFrame:
    """One row per trading day that produced a breakout."""
    df = df.copy()
    df["date"] = df["ts_minute"].dt.date
    df["t"] = df["ts_minute"].dt.time

    # Prior-session close (last bar of the previous day) -> gap.
    eod_close = df.groupby("date")["close"].last()
    prev_close = eod_close.shift(1)

    or_end = (pd.Timestamp.combine(pd.Timestamp.today(), SESSION_OPEN)
              + pd.Timedelta(minutes=or_minutes)).time()

    rows = []
    for d, g in df.groupby("date", sort=True):
        g = g.sort_values("ts_minute").reset_index(drop=True)
        opening = g[(g["t"] >= SESSION_OPEN) & (g["t"] < or_end)]
        if len(opening) < or_minutes:
            continue  # short/odd session, skip
        or_high = opening["high"].max()
        or_low = opening["low"].min()
        day_open = opening["open"].iloc[0]

        after = g[g["t"] >= or_end].reset_index(drop=True)
        if after.empty:
            continue

        # First bar that pierces either boundary.
        up = after["high"] >= or_high
        dn = after["low"] <= or_low
        i_up = up.idxmax() if up.any() else None
        i_dn = dn.idxmax() if dn.any() else None
        if i_up is None and i_dn is None:
            continue  # inside day, no signal

        if i_dn is None or (i_up is not None and i_up < i_dn):
            direction, entry_px, i_brk = 1, or_high, i_up
        elif i_up is None or i_dn < i_up:
            direction, entry_px, i_brk = -1, or_low, i_dn
        else:  # same bar pierces both -> classify by that bar's body
            bar = after.iloc[i_up]
            direction = 1 if bar["close"] >= bar["open"] else -1
            entry_px = or_high if direction == 1 else or_low
            i_brk = i_up

        j = min(i_brk + fwd_min, len(after) - 1)
        fwd_px = after["close"].iloc[j]
        eod_px = g["close"].iloc[-1]

        pc = prev_close.get(d, np.nan)
        rows.append({
            "date": d,
            "direction": direction,
            "or_width_bps": (or_high - or_low) / or_low * 1e4,
            "gap_bps": (day_open / pc - 1) * 1e4 if pd.notna(pc) else np.nan,
            "brk_time": after["ts_minute"].iloc[i_brk].time(),
            "fwd_bps": direction * (fwd_px / entry_px - 1) * 1e4,
            "eod_bps": direction * (eod_px / entry_px - 1) * 1e4,
        })

    return pd.DataFrame(rows)


def _summarise(grp: pd.DataFrame) -> dict:
    return {
        "n": len(grp),
        "hit_fwd": (grp["fwd_bps"] > 0).mean() * 100,
        "med_fwd_bps": grp["fwd_bps"].median(),
        "mean_fwd_bps": grp["fwd_bps"].mean(),
        "hit_eod": (grp["eod_bps"] > 0).mean() * 100,
        "mean_eod_bps": grp["eod_bps"].mean(),
    }


def _bucket_table(df: pd.DataFrame, col: str, labels: list[str]) -> pd.DataFrame:
    binned = pd.qcut(df[col], q=len(labels), labels=labels, duplicates="drop")
    rows = []
    for label, grp in df.groupby(binned, observed=True):
        r = {col + "_bin": label, f"{col}_median": grp[col].median()}
        r.update(_summarise(grp))
        rows.append(r)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, *, fwd_min: int = FWD_MIN) -> None:
    df = df.dropna(subset=["fwd_bps", "eod_bps"]).copy()
    n_long = int((df["direction"] == 1).sum())
    n_short = int((df["direction"] == -1).sum())

    print(f"\nORB breakout — {len(df)} signal-days  "
          f"(long {n_long} / short {n_short})")
    print("Signed forward return: + = breakout continued, - = it faded.")
    o = _summarise(df)
    print("=" * 78)
    print(f"OVERALL   hit@{fwd_min}m {o['hit_fwd']:.1f}%   "
          f"mean@{fwd_min}m {o['mean_fwd_bps']:+.2f} bps   "
          f"median {o['med_fwd_bps']:+.2f} bps   |   "
          f"hit@EOD {o['hit_eod']:.1f}%   mean@EOD {o['mean_eod_bps']:+.2f} bps")

    print(f"\nBy opening-range width quintile  (narrow=range day, wide=trend day)")
    print("-" * 78)
    print(_bucket_table(df, "or_width_bps",
                        ["Q1_narrow", "Q2", "Q3", "Q4", "Q5_wide"]
                        ).to_string(index=False, float_format="%.2f"))

    if df["gap_bps"].notna().sum() >= 50:
        print(f"\nBy overnight gap quintile  (§4: large gaps break ORB)")
        print("-" * 78)
        print(_bucket_table(df.dropna(subset=["gap_bps"]), "gap_bps",
                            ["Q1_gapdn", "Q2", "Q3_flat", "Q4", "Q5_gapup"]
                            ).to_string(index=False, float_format="%.2f"))


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    spot = load_spot(con)
    print(f"Loaded {len(spot):,} spot 1-min bars  "
          f"({spot['ts_minute'].min()} → {spot['ts_minute'].max()})")
    sig = compute_orb_signals(spot)
    report(sig)
    con.close()


if __name__ == "__main__":
    main()
