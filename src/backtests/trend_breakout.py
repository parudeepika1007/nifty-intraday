"""Backtest: opening-range trend-day breakout, held to the close.

Why this one (after 3 scalp/reversal rejections): an option BUYER needs a
big, sustained move to outrun theta. So instead of scalping, this takes at
most ONE trade a day — a break of the opening range on a likely trend day —
and HOLDS IT TO THE CLOSE, trying to capture the whole directional day.
Few trades, fat winners, convexity-friendly.

Logic. The first 30 min (09:15-09:45) sets the range. A clean break of it,
on a day whose opening range is wide enough to suggest a trend day (our ORB
study: widest opening ranges continued 58.6% of the time), tends to run.

Signal (per day):
  OR = high/low of 09:15-09:45.  OR-width percentile vs the trailing 60 days
  must be >= WIDTH_PCT (wide = trend-prone; skip narrow range days).
  First 1-min CLOSE beyond OR after 09:45 (and before ENTRY_LAST) = entry.
  Long on up-break, short on down-break. Stop = the far side of the OR (1R).
  Exit at 15:20 (hold the whole day) or if the stop is hit.
"""
from __future__ import annotations

import datetime as dt

import duckdb
import numpy as np
import pandas as pd

from ..schema import DB_PATH
from . import _report as rep

OR_START, OR_END = dt.time(9, 15), dt.time(9, 45)
ENTRY_LAST = dt.time(14, 0)
SQUARE_OFF = dt.time(15, 20)
WIDTH_PCT = 0.40        # OR width must be >= 40th percentile of last 60 days
OPT_THETA_RS = 15.0     # ATM weekly theta bled over a full-day hold (non-expiry)


def load_spot(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.sql("""SELECT ts_minute, high AS hi, low AS lo, close
                    FROM bars_1min WHERE symbol='NIFTY_SPOT'
                    ORDER BY ts_minute""").df()
    for c in ("hi", "lo", "close"):
        df[c] = df[c].astype(float)
    df["date"] = df["ts_minute"].dt.date
    df["t"] = df["ts_minute"].dt.time
    return df


def simulate(df: pd.DataFrame) -> pd.DataFrame:
    days = list(df.groupby("date", sort=True))
    or_widths: list[float] = []          # trailing OR widths for the percentile
    trades = []
    for d, g in days:
        g = g.reset_index(drop=True)
        opening = g[(g["t"] >= OR_START) & (g["t"] < OR_END)]
        if len(opening) < 20:
            continue
        orh, orl = opening["hi"].max(), opening["lo"].min()
        width = orh - orl
        # percentile of this width vs the trailing 60 days (then record it)
        pct = (np.mean([w < width for w in or_widths[-60:]])
               if len(or_widths) >= 20 else 1.0)
        or_widths.append(width)
        if pct < WIDTH_PCT or width <= 0:
            continue

        after = g[(g["t"] >= OR_END) & (g["t"] <= ENTRY_LAST)].reset_index(drop=True)
        if after.empty:
            continue
        up = after["close"] > orh
        dn = after["close"] < orl
        iu = up.idxmax() if up.any() else None
        idn = dn.idxmax() if dn.any() else None
        if iu is None and idn is None:
            continue
        if idn is None or (iu is not None and iu < idn):
            side, k, entry, stop = 1, iu, after["close"].iloc[iu], orl
        else:
            side, k, entry, stop = -1, idn, after["close"].iloc[idn], orh
        risk = abs(entry - stop)
        if risk <= 0:
            continue

        rest = g[g["ts_minute"] >= after["ts_minute"].iloc[k]].reset_index(drop=True)
        exit_px = rest["close"].iloc[-1]
        for _, b in rest.iterrows():
            if b["t"] >= SQUARE_OFF:
                exit_px = b["close"]; break
            if side == 1 and b["lo"] <= stop:
                exit_px = stop; break
            if side == -1 and b["hi"] >= stop:
                exit_px = stop; break
        R = ((exit_px - entry) if side == 1 else (entry - exit_px)) / risk
        trades.append({"date": d, "side": "long" if side == 1 else "short",
                       "R": R, "risk_pts": risk})
    return pd.DataFrame(trades)


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = load_spot(con)
    con.close()
    tr = simulate(df)
    rule = [
        "Take the first break of the 09:15-09:45 opening range, but only on",
        "days whose range is wide (>= 40th pct of last 60 days = trend-prone).",
        "Long on up-break / short on down-break. Stop = far side of the range.",
        "HOLD TO 15:20 to capture the whole trend day (few trades, big winners).",
    ]
    rep.render(
        "OPENING-RANGE TREND-DAY BREAKOUT  —  BACKTEST REPORT", rule, tr,
        n_signals=len(tr), n_days=df["date"].nunique(),
        period=(df["ts_minute"].min().date(), df["ts_minute"].max().date()),
        hold_desc="to the close (~all day)", opt_theta_rs=OPT_THETA_RS)


if __name__ == "__main__":
    main()
