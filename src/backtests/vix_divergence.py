"""Backtest: NIFTY-VIX divergence reversal (idea #4 / memo S2).

Market logic. NIFTY and India VIX are structurally anti-correlated (~-0.8):
price up, fear down. When they break that and move the SAME direction
intraday, it flags institutional hedging / repositioning that usually
precedes a REVERSAL over the next 1-3 hours:
  * NIFTY up  AND VIX up   -> fear rising into a rally  -> fade it (SHORT)
  * NIFTY down AND VIX down -> complacency into a drop   -> fade it (LONG)
Both are indices, so this is fully backtestable on 2 yr of 1-min data.
The 1-3h reversal is a big, slow move that suits OPTION BUYING (theta has
time to be outrun), unlike the rejected 3-min scalp.

Signal (per 1-min bar, reset each session):
  nifty_mom = % change over the last LOOKBACK min
  vix_mom   = % change over the last LOOKBACK min
  enter SHORT when both > 0 with vix_mom >= VIX_MOVE_MIN and nifty_mom >= NIFTY_MOVE_MIN
  enter LONG  when both < 0 with the mirror thresholds
Entry next bar; stop STOP_PCT away (= 1R); exit after HOLD_MIN or 15:20.
"""
from __future__ import annotations

import datetime as dt

import duckdb
import numpy as np
import pandas as pd

from ..schema import DB_PATH
from . import _report as rep

LOOKBACK = 15          # minutes for the co-move momentum
VIX_MOVE_MIN = 0.50    # VIX must have moved >= 0.50% over the lookback
NIFTY_MOVE_MIN = 0.10  # NIFTY must have moved >= 0.10%
HOLD_MIN = 120         # hold up to 2 hours (the reversal window)
STOP_PCT = 0.40        # adverse % stop = 1R
ENTRY_LAST = dt.time(13, 30)   # last entry so the 2h hold fits the session
OPT_THETA_RS = 8.0     # ~2h of ATM weekly theta (non-expiry), premium rupees


def load_aligned(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.sql("""
        SELECT s.ts_minute, s.high AS hi, s.low AS lo, s.close AS spot,
               v.close AS vix
        FROM bars_1min s
        JOIN bars_1min v ON v.ts_minute = s.ts_minute AND v.symbol = 'INDIA_VIX'
        WHERE s.symbol = 'NIFTY_SPOT'
        ORDER BY s.ts_minute
    """).df()
    for c in ("hi", "lo", "spot", "vix"):
        df[c] = df[c].astype(float)
    df["date"] = df["ts_minute"].dt.date
    df["t"] = df["ts_minute"].dt.time
    return df


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # momentum over LOOKBACK, only valid when the lookback bar is the same day
    df["spot_back"] = df["spot"].shift(LOOKBACK)
    df["vix_back"] = df["vix"].shift(LOOKBACK)
    df["date_back"] = df["date"].shift(LOOKBACK)
    same = df["date_back"] == df["date"]
    df["nifty_mom"] = np.where(same, df["spot"] / df["spot_back"] - 1, np.nan)
    df["vix_mom"] = np.where(same, df["vix"] / df["vix_back"] - 1, np.nan)
    nm, vm = df["nifty_mom"], df["vix_mom"]
    df["sig"] = 0
    df.loc[(nm >= NIFTY_MOVE_MIN / 100) & (vm >= VIX_MOVE_MIN / 100), "sig"] = -1  # fade up
    df.loc[(nm <= -NIFTY_MOVE_MIN / 100) & (vm <= -VIX_MOVE_MIN / 100), "sig"] = 1  # fade down
    # forward 60-min NIFTY return (study only)
    df["fwd60"] = df["spot"].shift(-60) / df["spot"] - 1
    fwd_same = df["date"].shift(-60) == df["date"]
    df.loc[~fwd_same, "fwd60"] = np.nan
    return df


def simulate(df: pd.DataFrame) -> pd.DataFrame:
    spot, hi, lo = df["spot"].values, df["hi"].values, df["lo"].values
    sig, date, tm = df["sig"].values, df["date"].values, df["t"].values
    n = len(df)
    trades, i = [], 0
    while i < n - 1:
        s = sig[i]
        if s == 0 or tm[i] > ENTRY_LAST:
            i += 1
            continue
        e = i + 1
        if date[e] != date[i]:
            i += 1
            continue
        entry = spot[e]
        stop = entry * (1 - STOP_PCT / 100) if s == 1 else entry * (1 + STOP_PCT / 100)
        risk = abs(entry - stop)
        exit_px, j = None, e
        while j < n and date[j] == date[e]:
            last_of_day = (j + 1 >= n) or (date[j + 1] != date[e])
            if s == 1:
                if lo[j] <= stop:
                    exit_px = stop; break
            else:
                if hi[j] >= stop:
                    exit_px = stop; break
            if j >= e + HOLD_MIN or tm[j] >= dt.time(15, 20) or last_of_day:
                exit_px = spot[j]; break
            j += 1
        if exit_px is None:
            exit_px = spot[min(j, n - 1)]
        R = ((exit_px - entry) if s == 1 else (entry - exit_px)) / risk
        trades.append({"date": date[e], "side": "long" if s == 1 else "short",
                       "R": R, "risk_pts": risk})
        i = j + 1
    return pd.DataFrame(trades)


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = build_signals(load_aligned(con))
    con.close()

    # hypothesis check: does forward 60-min NIFTY move reverse the co-move?
    up = df[df["sig"] == -1]["fwd60"].dropna() * 10000   # both-up -> expect down
    dn = df[df["sig"] == 1]["fwd60"].dropna() * 10000     # both-down -> expect up
    base = df["fwd60"].dropna() * 10000
    extra = [
        "CO-MOVE FORWARD STUDY (raw, all qualifying minutes):",
        f"  both UP  (fade->short): fwd60 mean {up.mean():+.1f} bps  (n={len(up)})  "
        f"-> want NEGATIVE",
        f"  both DOWN (fade->long): fwd60 mean {dn.mean():+.1f} bps  (n={len(dn)})  "
        f"-> want POSITIVE",
        f"  baseline all-minutes : fwd60 mean {base.mean():+.1f} bps",
    ]

    tr = simulate(df)
    rule = [
        "When NIFTY and India VIX move the SAME way over 15 min (breaking their",
        "normal inverse link), FADE it: both-up -> short, both-down -> long.",
        f"Need VIX move >= {VIX_MOVE_MIN}% and NIFTY move >= {NIFTY_MOVE_MIN}%.",
        f"Enter next bar; stop {STOP_PCT}% (=1R); hold up to {HOLD_MIN}min or 15:20.",
    ]
    n_days = df["date"].nunique()
    rep.render(
        "NIFTY-VIX DIVERGENCE REVERSAL  —  BACKTEST REPORT", rule, tr,
        n_signals=int((df["sig"] != 0).sum()), n_days=n_days,
        period=(df["ts_minute"].min().date(), df["ts_minute"].max().date()),
        hold_desc=f"up to {HOLD_MIN} min (option-buyer friendly)",
        opt_theta_rs=OPT_THETA_RS, extra_lines=extra)


if __name__ == "__main__":
    main()
