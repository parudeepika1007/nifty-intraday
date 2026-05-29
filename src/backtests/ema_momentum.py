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
# extension cap (the S5 climax-fade fix): skip entries that are already late
EXT_MAX = 2.0        # close must be within 2*ATR of EMA20 (not over-extended)
BODY_ATR_MAX = 2.5   # skip blow-off candles bigger than 2.5*ATR
COST_POINTS = 2.0    # assumed futures round-trip cost (slippage+fees) in index pts
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


def add_features(df: pd.DataFrame, *, ext_max: float | None = None,
                 body_atr_max: float | None = None) -> pd.DataFrame:
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
    df["ext20"] = (c - df["ema20"]) / df["atr"]   # how far close sits from EMA20
    strong_bull = ((body > 0) & (df["body_ratio"] >= BODY_RATIO_MIN)
                   & (df["body_atr"] >= BODY_ATR_MIN) & (df["close_up"] >= CLOSE_POS_MIN))
    strong_bear = ((body < 0) & (df["body_ratio"] >= BODY_RATIO_MIN)
                   & (df["body_atr"] >= BODY_ATR_MIN) & (df["close_dn"] >= CLOSE_POS_MIN))
    intraday = (df["t"] >= ENTRY_START) & (df["t"] <= ENTRY_END)

    if body_atr_max is not None:                  # drop blow-off climax candles
        ok = df["body_atr"] <= body_atr_max
        strong_bull &= ok
        strong_bear &= ok
    not_ext_bull = (df["ext20"] <= ext_max) if ext_max is not None else True
    not_ext_bear = (-df["ext20"] <= ext_max) if ext_max is not None else True

    df["sig"] = 0
    df.loc[gate_bull & strong_bull & (tag_lo <= df["ema20"]) & intraday
           & not_ext_bull, "sig"] = 1
    df.loc[gate_bear & strong_bear & (tag_hi >= df["ema20"]) & intraday
           & not_ext_bear, "sig"] = -1
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
                       "R": R, "risk_pts": risk,
                       "body_atr": body_atr[i], "fan": abs(fan[i])})
        i = j + 1
    return pd.DataFrame(trades)


def stats(tr: pd.DataFrame, cost_points: float = 0.0) -> dict:
    if tr.empty:
        return {"n": 0, "win%": 0, "expR": 0, "totR": 0, "pf": 0}
    r = tr["R"] - (cost_points / tr["risk_pts"] if cost_points else 0.0)
    wins, losses = r[r > 0], r[r <= 0]
    gl = -losses.sum()
    return {"n": len(r), "win%": len(wins) / len(r) * 100,
            "expR": r.mean(), "totR": r.sum(),
            "pf": (wins.sum() / gl) if gl > 0 else float("inf")}


def _row(label: str, st: dict) -> str:
    return (f"{label:14} n={st['n']:5d}  win={st['win%']:5.1f}%  "
            f"exp={st['expR']:+.3f}R  PF={st['pf']:4.2f}  total={st['totR']:+7.1f}R")


def _oos(tr: pd.DataFrame, cost: float) -> None:
    days = np.sort(tr["date"].unique())
    cut = days[int(len(days) * 0.8)]
    ins, oos = tr[tr["date"] < cut], tr[tr["date"] >= cut]
    print(f"  walk-forward (cut {cut}, net):")
    print("   " + _row("in-sample", stats(ins, cost)))
    print("   " + _row("out-sample", stats(oos, cost)))


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    base_df = load_3min(con)
    con.close()
    base = add_features(base_df.copy())
    print(f"3-min bars: {len(base):,}  ({base['ts_minute'].min()} -> {base['ts_minute'].max()})")

    # 1) baseline exit-style comparison (gross)
    print(f"\nBASELINE  signals={(base['sig']!=0).sum()}  "
          f"(long {(base['sig']==1).sum()} / short {(base['sig']==-1).sum()})")
    print("Exit-style comparison (gross, 1R = risk to stop):")
    print("=" * 70)
    results = {st: simulate(base, st) for st in ("r2_ema", "ema_trail", "timebox")}
    for st, tr in results.items():
        print(_row(st, stats(tr)))
    best = max((k for k in results if len(results[k]) >= 100),
               key=lambda k: stats(results[k])["expR"])
    print(f"-> best exit: '{best}'")

    # 2) extension-capped variant on the best exit
    capped = add_features(base_df.copy(), ext_max=EXT_MAX, body_atr_max=BODY_ATR_MAX)
    tr_b, tr_c = results[best], simulate(capped, best)
    print(f"\nEXTENSION CAP  (close within {EXT_MAX}*ATR of EMA20, body<= {BODY_ATR_MAX}*ATR)")
    print(f"signals={(capped['sig']!=0).sum()}  on exit '{best}':")
    print("=" * 70)
    print(_row("baseline gross", stats(tr_b)))
    print(_row("capped  gross", stats(tr_c)))
    print(_row(f"baseline net@{COST_POINTS}", stats(tr_b, COST_POINTS)))
    print(_row(f"capped  net@{COST_POINTS}", stats(tr_c, COST_POINTS)))

    # 3) cost sensitivity on the capped variant
    print(f"\nCost sensitivity (capped, '{best}'):")
    print("-" * 70)
    for cp in (0.0, 1.0, 2.0, 3.0, 4.0):
        print(_row(f"cost {cp:.0f} pts", stats(tr_c, cp)))

    # 4) per-side + strength quintile + OOS, all NET of cost, capped
    print(f"\nCapped, net @ {COST_POINTS} pts:")
    print("-" * 70)
    for side in ("long", "short"):
        print(_row(side, stats(tr_c[tr_c["side"] == side], COST_POINTS)))
    g = tr_c.copy()
    g["strength"] = g["body_atr"] * g["fan"]
    g["q"] = pd.qcut(g["strength"], 5, labels=["S1", "S2", "S3", "S4", "S5"],
                     duplicates="drop")
    print("  strength quintile (body_atr x fan):")
    for q, gg in g.groupby("q", observed=True):
        print("   " + _row(str(q), stats(gg, COST_POINTS)))
    print()
    _oos(tr_c, COST_POINTS)


if __name__ == "__main__":
    main()
