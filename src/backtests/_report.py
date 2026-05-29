"""Shared, readable backtest report — used by every strategy so results are
always presented the same plain-English way (the user's standing preference).

A strategy hands over a trades DataFrame with columns:
    date, side ('long'/'short'), R (result in risk-multiples), risk_pts (stop
    distance in index points).
render() then prints: the rule, gross result, net of FUTURES cost, net of
OPTION-BUYING cost (ATM/ITM), out-of-sample robustness, a verdict, and a
glossary. Costs are in index-equivalent points and scale by each trade's stop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COST_POINTS = 2.0     # futures round-trip slippage+fees, index points
OPT_SPREAD_RS = 2.0   # option bid-ask crossed in+out, premium rupees
OPT_FEES_RS = 1.0     # brokerage+taxes per share, premium rupees
ATM_DELTA = 0.50
ITM_DELTA = 0.70


def stats(tr: pd.DataFrame, cost_points: float = 0.0) -> dict:
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


def opt_cost_pts(delta: float, theta_rs: float) -> float:
    """Option-buying friction in INDEX-equivalent points.
    A delta-D option captures only D of the move, so matching one futures'
    exposure costs friction/D, where friction = spread+fees+theta over the hold.
    """
    return (OPT_SPREAD_RS + OPT_FEES_RS + theta_rs) / delta


def line(label: str, st: dict) -> str:
    pf = " inf" if st["pf"] == float("inf") else f"{st['pf']:.2f}"
    return (f"  {label:26} trades {st['n']:4d} | win {st['win']:5.1f}% | "
            f"avg {st['exp']:+.3f}R | PF {pf:>4} | total {st['tot']:+7.1f}R")


def render(title: str, rule_lines: list[str], tr: pd.DataFrame, *,
           n_signals: int, n_days: int, period: tuple, hold_desc: str,
           opt_theta_rs: float, extra_lines: list[str] | None = None) -> None:
    bar = "=" * 74
    print(bar)
    print(f"  {title}")
    print(bar)
    print(f"  Period   : {period[0]}  ->  {period[1]}   ({n_days} trading days)")
    print(f"  Signals  : {n_signals}   (~{n_signals/max(n_days,1):.2f} per day)   "
          f"hold: {hold_desc}")
    print()
    print("  THE RULE (plain English)")
    for ln in rule_lines:
        print(f"    {ln}")
    print()

    if extra_lines:
        print("  " + "-" * 70)
        for ln in extra_lines:
            print(f"  {ln}")
        print()

    g = stats(tr)
    print("  " + "-" * 70)
    print("  GROSS RESULT  (before any trading costs)")
    print("  " + "-" * 70)
    print(line("ALL trades", g))
    if g["n"]:
        print(f"    avg winner {g['aw']:+.2f}R   avg loser {g['al']:+.2f}R   "
              f"best {g['best']:+.1f}R   worst {g['worst']:+.1f}R")
    print(line("long only", stats(tr[tr.side == "long"])))
    print(line("short only", stats(tr[tr.side == "short"])))
    print()

    print("  " + "-" * 70)
    print("  NET OF COSTS  —  what you actually keep")
    print("  " + "-" * 70)
    print("  Trading FUTURES (round-trip ~2 index pts):")
    print(line("futures, net", stats(tr, COST_POINTS)))
    fric = OPT_SPREAD_RS + OPT_FEES_RS + opt_theta_rs
    atm, itm = opt_cost_pts(ATM_DELTA, opt_theta_rs), opt_cost_pts(ITM_DELTA, opt_theta_rs)
    print(f"  BUYING OPTIONS: friction Rs{fric:.0f} (spread {OPT_SPREAD_RS:.0f}+fees "
          f"{OPT_FEES_RS:.0f}+theta {opt_theta_rs:.0f} over hold):")
    print(f"    ATM delta {ATM_DELTA:.2f} -> {atm:.1f} index-equiv pts")
    print(line("ATM option, net", stats(tr, atm)))
    print(f"    ITM delta {ITM_DELTA:.2f} -> {itm:.1f} index-equiv pts")
    print(line("ITM option, net", stats(tr, itm)))
    print()

    if g["n"] >= 20:
        d = np.sort(tr["date"].unique())
        cut = d[int(len(d) * 0.8)]
        print("  " + "-" * 70)
        print("  ROBUSTNESS  (net of futures cost, 80/20 split by date)")
        print("  " + "-" * 70)
        print(line(f"in-sample  (< {cut})", stats(tr[tr.date < cut], COST_POINTS)))
        print(line(f"out-sample (>= {cut})", stats(tr[tr.date >= cut], COST_POINTS)))
        print()

    fut = stats(tr, COST_POINTS)
    print(bar)
    print("  VERDICT")
    if fut["exp"] > 0.02 and fut["pf"] > 1.10 and fut["n"] >= 100:
        print("  Positive net edge after futures cost — worth refining / forward-testing.")
        if stats(tr, opt_cost_pts(ITM_DELTA, opt_theta_rs))["exp"] > 0:
            print("  Survives ITM option-buying cost too — promising for your style.")
        else:
            print("  But option-buying cost erases it — would need futures, or bigger moves.")
    else:
        print("  REJECTED. Net edge after costs is not there "
              f"(futures net {fut['exp']:+.3f}R, PF {fut['pf']:.2f}).")
    print(bar)
    print("  GLOSSARY:  R = risk multiples (-1R = stop). win% = profitable trades.")
    print("  avg R = expectancy per trade (the edge). PF = wins/losses (>1 = profit).")
    print(bar)
