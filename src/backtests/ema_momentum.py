"""Backtest: 3-min EMA-ribbon momentum (long & short).

Idea (user's chart read): trade a STRONG conviction candle only when the
EMA ribbon is already inclined AND fanned out in the same direction. When
the ribbon is braided/flat, the same candle is noise — so the regime gate
is the whole game.

  Gate (trend regime, the filter):
    bull = EMA10>EMA20>EMA50  AND  EMA20 inclined up  AND  ribbon fanned
    bear = mirror.   slope and fan are normalised by ATR so thresholds are
    scale-free.  Anything else = NO-TRADE (the braided-ribbon chop).

  Trigger (strong candle, resume after a pullback to the ribbon):
    body/range >= 0.6, body >= 1.2*ATR, close in the top/bottom 30% of its
    range, price tagged the EMA20 within the last 3 bars, closes the trend way.

  Entry next bar's open. Stop = trigger candle extreme (or 1.2*ATR floor).
  Three EXIT styles are simulated and compared; the data picks the winner:
    r2_ema   : +2R target, or close back through EMA20, or 15:20 square-off
    ema_trail: ride until a close back through EMA20, or 15:20 square-off
    timebox  : exit 30 min (10 bars) after entry, or 15:20 square-off

Notes / honesty:
  * 3-min bars are resampled from the 1-min spot table, anchored to 09:15.
  * EMAs/ATR run continuously across days (matches how the ribbon is drawn).
  * NIFTY spot 1-min volume is unusable (0 on ~72% of bars), so "strong" is
    defined on PRICE only (body/ATR, body/range) — no volume confirmation.
  * Spot isn't tradable; this measures the DIRECTIONAL edge. Real fills are
    in futures/options and will differ by slippage/cost.
"""
from __future__ import annotations

import datetime as dt

import duckdb
import numpy as np
import pandas as pd

from ..schema import DB_PATH

# --- gate / trigger thresholds (tunable; defaults are deliberate, not fitted)
SLOPE_MIN = 0.25   # EMA20 must move >= 0.25*ATR over the last 5 bars
FAN_MIN = 0.50     # (EMA10-EMA50) must span >= 0.50*ATR
BODY_ATR_MIN = 1.2
BODY_RATIO_MIN = 0.60
CLOSE_POS_MIN = 0.70
ENTRY_START = dt.time(9, 30)
ENTRY_END = dt.time(15, 0)
SQUARE_OFF = dt.time(15, 20)
TIMEBOX_BARS = 10  # 10 * 3min = 30 min


def load_3min(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df1 = con.sql("""SELECT ts_minute, open, high, low, close
                     FROM bars_1min WHERE symbol='NIFTY_SPOT'
                     ORDER BY ts_minute""").df()
    for c in ("open", "high", "low", "close"):
        df1[c] = df1[c].astype(float)
    df1["date"] = df1["ts_minute"].dt.date
    out = []
    for _, g in df1.groupby("date", sort=True):
        g = g.reset_index(drop=True)
        grp = g.index // 3
        out.append(g.groupby(grp).agg(
            ts_minute=("ts_minute", "first"), open=("open", "first"),
            high=("high", "max"), low=("low", "min"), close=("close", "last")))
    df3 = pd.concat(out, ignore_index=True)
    df3["date"] = df3["ts_minute"].dt.date
    df3["t"] = df3["ts_minute"].dt.time
    return df3


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    for n in (10, 20, 50, 200):
        df[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    firstbar = df["date"] != df["date"].shift(1)
    tr[firstbar] = (h - l)[firstbar]
    df["atr"] = tr.rolling(14, min_periods=7).mean()

    df["slope20"] = (df["ema20"] - df["ema20"].shift(5)) / df["atr"]
    df["fan"] = (df["ema10"] - df["ema50"]) / df["atr"]
    rng = (h - l).replace(0, np.nan)
    body = c - o
    df["body_atr"] = body.abs() / df["atr"]
    df["body_ratio"] = body.abs() / rng
    df["close_up"] = (c - l) / rng
    df["close_dn"] = (h - c) / rng

    bull_stack = (df["ema10"] > df["ema20"]) & (df["ema20"] > df["ema50"])
    bear_stack = (df["ema10"] < df["ema20"]) & (df["ema20"] < df["ema50"])
    gate_bull = bull_stack & (df["slope20"] >= SLOPE_MIN) & (df["fan"] >= FAN_MIN)
    gate_bear = bear_stack & (df["slope20"] <= -SLOPE_MIN) & (df["fan"] <= -FAN_MIN)

    tag_lo = l.rolling(3).min()
    tag_hi = h.rolling(3).max()
    strong_bull = ((body > 0) & (df["body_ratio"] >= BODY_RATIO_MIN)
                   & (df["body_atr"] >= BODY_ATR_MIN) & (df["close_up"] >= CLOSE_POS_MIN))
    strong_bear = ((body < 0) & (df["body_ratio"] >= BODY_RATIO_MIN)
                   & (df["body_atr"] >= BODY_ATR_MIN) & (df["close_dn"] >= CLOSE_POS_MIN))
    intraday = (df["t"] >= ENTRY_START) & (df["t"] <= ENTRY_END)

    df["sig"] = 0
    df.loc[gate_bull & strong_bull & (tag_lo <= df["ema20"]) & intraday, "sig"] = 1
    df.loc[gate_bear & strong_bear & (tag_hi >= df["ema20"]) & intraday, "sig"] = -1
    return df


def simulate(df: pd.DataFrame, exit_style: str) -> pd.DataFrame:
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    ema20, atr, sig = df["ema20"].values, df["atr"].values, df["sig"].values
    date, tm = df["date"].values, df["t"].values
    body_atr, fan = df["body_atr"].values, df["fan"].values
    n = len(df)
    trades = []
    i = 0
    while i < n - 1:
        s = sig[i]
        if s == 0 or np.isnan(atr[i]) or atr[i] <= 0:
            i += 1
            continue
        e = i + 1
        if date[e] != date[i]:
            i += 1
            continue
        entry = o[e]
        if s == 1:
            stop = l[i]
            if entry - stop < 0.3 * atr[i]:
                stop = entry - 1.2 * atr[i]
            risk = entry - stop
            target = entry + 2 * risk
        else:
            stop = h[i]
            if stop - entry < 0.3 * atr[i]:
                stop = entry + 1.2 * atr[i]
            risk = stop - entry
            target = entry - 2 * risk
        if risk <= 0:
            i += 1
            continue

        exit_px, j = None, e
        while j < n and date[j] == date[e]:
            last_of_day = (j + 1 >= n) or (date[j + 1] != date[e])
            eod = tm[j] >= SQUARE_OFF or last_of_day
            if s == 1:
                if l[j] <= stop:
                    exit_px = stop; break
                if exit_style == "r2_ema" and h[j] >= target:
                    exit_px = target; break
                if exit_style in ("r2_ema", "ema_trail") and c[j] < ema20[j]:
                    exit_px = c[j]; break
                if exit_style == "timebox" and j >= e + TIMEBOX_BARS:
                    exit_px = c[j]; break
                if eod:
                    exit_px = c[j]; break
            else:
                if h[j] >= stop:
                    exit_px = stop; break
                if exit_style == "r2_ema" and l[j] <= target:
                    exit_px = target; break
                if exit_style in ("r2_ema", "ema_trail") and c[j] > ema20[j]:
                    exit_px = c[j]; break
                if exit_style == "timebox" and j >= e + TIMEBOX_BARS:
                    exit_px = c[j]; break
                if eod:
                    exit_px = c[j]; break
            j += 1
        if exit_px is None:
            exit_px = c[min(j, n - 1)]
        R = ((exit_px - entry) if s == 1 else (entry - exit_px)) / risk
        trades.append({"date": date[e], "side": "long" if s == 1 else "short",
                       "R": R, "body_atr": body_atr[i], "fan": abs(fan[i])})
        i = j + 1
    return pd.DataFrame(trades)


def stats(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"n": 0, "win%": 0, "expR": 0, "totR": 0, "pf": 0}
    wins, losses = tr["R"][tr["R"] > 0], tr["R"][tr["R"] <= 0]
    gl = -losses.sum()
    return {"n": len(tr), "win%": len(wins) / len(tr) * 100,
            "expR": tr["R"].mean(), "totR": tr["R"].sum(),
            "pf": (wins.sum() / gl) if gl > 0 else float("inf")}


def _row(label: str, st: dict) -> str:
    return (f"{label:14} n={st['n']:5d}  win={st['win%']:5.1f}%  "
            f"exp={st['expR']:+.3f}R  PF={st['pf']:4.2f}  total={st['totR']:+7.1f}R")


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = add_features(load_3min(con))
    con.close()
    print(f"3-min bars: {len(df):,}  ({df['ts_minute'].min()} -> {df['ts_minute'].max()})")
    print(f"raw signals: {(df['sig']!=0).sum()}  "
          f"(long {(df['sig']==1).sum()} / short {(df['sig']==-1).sum()})\n")

    print("Exit-style comparison (1R = risk to stop):")
    print("=" * 70)
    results = {}
    for style in ("r2_ema", "ema_trail", "timebox"):
        tr = simulate(df, style)
        results[style] = tr
        print(_row(style, stats(tr)))
    print()

    # pick best by expectancy among styles with a usable sample
    usable = {k: v for k, v in results.items() if len(v) >= 100}
    best = max(usable or results, key=lambda k: stats(results[k])["expR"])
    tr = results[best]
    print(f"Best by expectancy: '{best}'")
    print("-" * 70)
    for side in ("long", "short"):
        print(_row(side, stats(tr[tr["side"] == side])))

    # confidence: does a stronger candle / wider fan grade better?
    tr = tr.copy()
    tr["strength"] = tr["body_atr"] * tr["fan"]
    tr["q"] = pd.qcut(tr["strength"], 5, labels=["S1", "S2", "S3", "S4", "S5"],
                      duplicates="drop")
    print("\nExpectancy by signal-strength quintile (body_atr x fan):")
    print("-" * 70)
    for q, g in tr.groupby("q", observed=True):
        print(_row(str(q), stats(g)))

    # walk-forward out-of-sample (80/20 by date)
    days = np.sort(tr["date"].unique())
    cut = days[int(len(days) * 0.8)]
    ins, oos = tr[tr["date"] < cut], tr[tr["date"] >= cut]
    print(f"\nWalk-forward (cut {cut}):")
    print("-" * 70)
    print(_row("in-sample", stats(ins)))
    print(_row("out-sample", stats(oos)))


if __name__ == "__main__":
    main()
