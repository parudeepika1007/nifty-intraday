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
COST_POINTS = 2.0    # assumed FUTURES round-trip cost (slippage+fees) in index pts
# option-buying cost model (premium rupees per share, round-trip over ~30-min hold)
OPT_SPREAD_RS = 2.0  # bid-ask paid crossing in and out (ATM weekly, non-expiry)
OPT_FEES_RS = 1.0    # brokerage + STT + exchange + GST, per share
OPT_THETA_RS = 2.0   # time-decay bled over the ~30-min hold (non-expiry ATM)
ATM_DELTA = 0.50     # an ATM option captures ~half the index move
ITM_DELTA = 0.70     # an ITM option captures ~0.7 of it
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
    """All results in R (multiples of risk). cost_points is charged per trade,
    scaled by each trade's stop distance (so it bites tight-stop trades harder)."""
    if tr.empty:
        return dict(n=0, win=0, exp=0, tot=0, pf=0, aw=0, al=0, best=0, worst=0)
    r = tr["R"] - (cost_points / tr["risk_pts"] if cost_points else 0.0)
    wins, losses = r[r > 0], r[r <= 0]
    gl = -losses.sum()
    return dict(n=len(r), win=len(wins) / len(r) * 100, exp=r.mean(), tot=r.sum(),
                pf=(wins.sum() / gl) if gl > 0 else float("inf"),
                aw=wins.mean() if len(wins) else 0.0,
                al=losses.mean() if len(losses) else 0.0,
                best=r.max(), worst=r.min())


def opt_cost_pts(delta: float) -> float:
    """Option-buying friction expressed in INDEX-equivalent points.

    Buying an option with delta D gives only D of the index exposure, so to
    match one futures' directional bet you carry 1/D options — and pay the
    premium friction (spread+fees+theta) on each. Hence the effective cost per
    unit of index exposure = friction / D. Theta and wide option spreads make
    this MUCH larger than the futures' ~2 pts — the key, counter-intuitive point.
    """
    return (OPT_SPREAD_RS + OPT_FEES_RS + OPT_THETA_RS) / delta


def line(label: str, st: dict) -> str:
    pf = " inf" if st["pf"] == float("inf") else f"{st['pf']:.2f}"
    return (f"  {label:24} trades {st['n']:4d} | win {st['win']:5.1f}% | "
            f"avg {st['exp']:+.3f}R | PF {pf:>4} | total {st['tot']:+7.1f}R")


def report(df: pd.DataFrame, tr: pd.DataFrame, best_exit: str,
           comparison: dict[str, pd.DataFrame]) -> None:
    days = df["date"].nunique()
    nsig = int((df["sig"] != 0).sum())
    bar = "=" * 74
    print(bar)
    print("  NIFTY 3-min EMA-RIBBON MOMENTUM  —  BACKTEST REPORT")
    print(bar)
    print(f"  Period   : {df['ts_minute'].min().date()}  ->  {df['ts_minute'].max().date()}   ({days} trading days)")
    print(f"  Data     : {len(df):,} three-min bars, resampled from 1-min NIFTY spot")
    print(f"  Signals  : {nsig}  (long {int((df['sig']==1).sum())}, short "
          f"{int((df['sig']==-1).sum())})   ~{nsig/days:.1f} per day")
    print()
    print("  THE RULE (plain English)")
    print("    Go LONG when the EMA ribbon (10>20>50) is stacked, EMA20 is sloping")
    print("    up, and the ribbon is fanned wide, AND a strong green candle closes")
    print("    after a small dip back to EMA20.  SHORT is the mirror.")
    print("    Enter next bar's open.  Stop = the candle's extreme (that risk = 1R).")
    print(f"    Exit after 30 min or 15:20 IST  ('{best_exit}' beat the other exits).")
    print()

    g = stats(tr)
    print("  " + "-" * 70)
    print("  GROSS RESULT  (before any trading costs)")
    print("  " + "-" * 70)
    print(line("ALL trades", g))
    print(f"    avg winner {g['aw']:+.2f}R   avg loser {g['al']:+.2f}R   "
          f"best {g['best']:+.1f}R   worst {g['worst']:+.1f}R")
    print(line("long only", stats(tr[tr.side == "long"])))
    print(line("short only", stats(tr[tr.side == "short"])))
    print()

    print("  " + "-" * 70)
    print("  NET OF COSTS  —  this is what you actually keep")
    print("  " + "-" * 70)
    print("  If trading FUTURES (round-trip ~2 index pts of slippage+fees):")
    print(line("futures, net", stats(tr, COST_POINTS)))
    print()
    fric = OPT_SPREAD_RS + OPT_FEES_RS + OPT_THETA_RS
    atm, itm = opt_cost_pts(ATM_DELTA), opt_cost_pts(ITM_DELTA)
    print("  If BUYING OPTIONS (ATM / ITM) — your intended way to trade:")
    print(f"    premium friction = Rs{OPT_SPREAD_RS:.0f} spread + Rs{OPT_FEES_RS:.0f} fees + "
          f"Rs{OPT_THETA_RS:.0f} theta(30min) = Rs{fric:.0f}/round-trip")
    print(f"    ATM delta {ATM_DELTA:.2f}  ->  effective cost = Rs{fric:.0f} / {ATM_DELTA:.2f} = "
          f"{atm:.1f} index-equiv pts")
    print(line("ATM option, net", stats(tr, atm)))
    print(f"    ITM delta {ITM_DELTA:.2f}  ->  effective cost = Rs{fric:.0f} / {ITM_DELTA:.2f} = "
          f"{itm:.1f} index-equiv pts")
    print(line("ITM option, net", stats(tr, itm)))
    print("    (note: this linear model ignores gamma/convexity, which would help")
    print("     the rare BIG fast winners; for tight 30-min scalps theta dominates.")
    print("     True option P&L needs historical option prices we don't have.)")
    print()

    print("  " + "-" * 70)
    print("  ROBUSTNESS  (net of futures cost, 80/20 split by date)")
    print("  " + "-" * 70)
    d = np.sort(tr["date"].unique())
    cut = d[int(len(d) * 0.8)]
    print(line(f"in-sample  (< {cut})", stats(tr[tr.date < cut], COST_POINTS)))
    print(line(f"out-sample (>= {cut})", stats(tr[tr.date >= cut], COST_POINTS)))
    print()

    print("  " + "-" * 70)
    print("  EXIT-STYLE COMPARISON  (gross, how we chose the exit)")
    print("  " + "-" * 70)
    for st, t in comparison.items():
        mark = "  <- chosen" if st == best_exit else ""
        print(line(st, stats(t)) + mark)
    print()

    fut = stats(tr, COST_POINTS)
    print(bar)
    print("  VERDICT")
    if fut["exp"] > 0.01 and fut["pf"] > 1.05:
        print("  Positive net edge on futures — worth refining / forward-testing.")
    else:
        print("  REJECTED. The gross edge is too thin to survive costs.")
        print(f"  Futures net: {fut['exp']:+.3f}R/trade (PF {fut['pf']:.2f}) — loses money.")
        print("  Option BUYING is WORSE, not better: theta + wide option spreads,")
        print(f"  divided by sub-1 delta, give ~{atm:.0f} index-pts of friction vs 2 for")
        print("  futures. Buying ATM/ITM does not rescue a weak directional signal.")
    print(bar)
    print("  WHAT THE NUMBERS MEAN")
    print("   R       multiples of risk.  -1R = stop hit.  +2R = made twice your risk.")
    print("   win%    share of trades that ended in profit.")
    print("   avg R   expectancy: average profit per trade in R.  THIS is the edge.")
    print("   PF      profit factor = total wins / total losses.  >1 makes money.")
    print("   total R sum of every trade's result — the 2-year P&L in risk-units.")
    print(bar)


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    base_df = load_3min(con)
    con.close()
    base = add_features(base_df.copy())
    comparison = {st: simulate(base, st) for st in ("r2_ema", "ema_trail", "timebox")}
    best = max((k for k in comparison if len(comparison[k]) >= 100),
               key=lambda k: stats(comparison[k])["exp"])
    report(base, comparison[best], best, comparison)


if __name__ == "__main__":
    main()
