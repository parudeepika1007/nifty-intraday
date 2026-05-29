"""Live monitor for ideas #2 (basis regime) and #3 (option net delta).

READ ONLY. This places NO orders — it polls live Dhan snapshots, computes
the two signals each tick, and prints a line. It exists so we can *watch*
the signals behave on live data, since neither can be backtested (their
historical contract IDs are gone; see CLAUDE.md §10 + memory).

  #2 basis      = front_fut_ltp - spot_ltp
     basis_z    = (basis - rolling mean) / rolling std   (window = --zwin)
                  The rolling window is SEEDED from today's 1-min bars so
                  z is meaningful from the very first live tick.
  #3 net_delta  = Σ_strikes (ce.delta * ce.oi + pe.delta * pe.oi)
                  Aggregate delta of all open option OI on the nearest
                  expiry. §3's signal is the *shift* in this, so we also
                  print the change since the previous tick.

Usage:
    python -m src.live.monitor                 # poll every 60s until close
    python -m src.live.monitor --interval 20 --iters 3
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from collections import deque
from datetime import date
from zoneinfo import ZoneInfo

import pandas as pd

from ..dhan import fetch_intraday
from ..instruments import NIFTY_SPOT, list_nifty_futures
from .feed import marketfeed_ltp, option_chain, option_expiry_list

IST = ZoneInfo("Asia/Kolkata")
NIFTY_SCRIP, NIFTY_SEG = 13, "IDX_I"


def resolve_front_future():
    today = date.today()
    live = [f for f in list_nifty_futures() if f.expiry >= today]
    if not live:
        raise RuntimeError("No live NIFTY future found in master CSV")
    return min(live, key=lambda f: f.expiry)


def seed_basis(front, zwin: int) -> deque:
    """Seed the rolling basis window from today's 1-min spot + future bars."""
    today = date.today()
    dq: deque = deque(maxlen=zwin)
    try:
        s = fetch_intraday(NIFTY_SPOT["security_id"], NIFTY_SEG, "INDEX", today, today)
        f = fetch_intraday(front.security_id, "NSE_FNO", "FUTIDX", today, today)
        if s.get("timestamp") and f.get("timestamp"):
            sp = pd.Series(s["close"], index=s["timestamp"])
            fu = pd.Series(f["close"], index=f["timestamp"])
            joined = pd.DataFrame({"s": sp, "f": fu}).dropna()
            for b in (joined["f"] - joined["s"]).tolist():
                dq.append(float(b))
    except Exception as e:  # seeding is best-effort; live ticks still work
        print(f"  (seed skipped: {e})")
    return dq


def nearest_expiry() -> str:
    expiries = sorted(option_expiry_list(NIFTY_SCRIP, NIFTY_SEG))
    today = date.today().isoformat()
    future = [e for e in expiries if e >= today]
    return future[0] if future else expiries[0]


def net_option_delta(expiry: str) -> tuple[float, float, float]:
    """Returns (net_delta, total_ce_oi, total_pe_oi) over the chain."""
    data = option_chain(expiry, NIFTY_SCRIP, NIFTY_SEG)
    net = ce_oi = pe_oi = 0.0
    for strike, leg in data.get("oc", {}).items():
        ce, pe = leg.get("ce") or {}, leg.get("pe") or {}
        if ce.get("oi"):
            net += (ce["greeks"]["delta"] or 0) * ce["oi"]
            ce_oi += ce["oi"]
        if pe.get("oi"):
            net += (pe["greeks"]["delta"] or 0) * pe["oi"]
            pe_oi += pe["oi"]
    return net, ce_oi, pe_oi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60, help="seconds between ticks")
    ap.add_argument("--iters", type=int, default=0, help="max ticks (0 = until close)")
    ap.add_argument("--zwin", type=int, default=60, help="basis z rolling window")
    args = ap.parse_args()

    front = resolve_front_future()
    expiry = nearest_expiry()
    print(f"Live monitor (READ ONLY — no orders)")
    print(f"  spot=NIFTY({NIFTY_SCRIP})  front_fut={front.symbol}({front.security_id})"
          f"  opt_expiry={expiry}  zwin={args.zwin}")
    basis_win = seed_basis(front, args.zwin)
    print(f"  seeded {len(basis_win)} basis obs from today's 1-min bars\n")

    segs = {NIFTY_SEG: [NIFTY_SCRIP], "NSE_FNO": [front.security_id]}
    prev_net = None
    n = 0
    print(f"{'time':8}  {'spot':>9} {'fut':>9} {'basis':>7} {'z':>6}  "
          f"{'netΔ·OI(mn)':>11} {'Δ since':>9}  {'PCR':>5}")
    print("-" * 74)
    while True:
        now = dt.datetime.now(IST)
        if now.time() > dt.time(15, 30):
            print("Market closed (>15:30 IST). Stopping.")
            break
        try:
            ltp = marketfeed_ltp(segs)
            spot = ltp[NIFTY_SEG][str(NIFTY_SCRIP)]["last_price"]
            fut = ltp["NSE_FNO"][str(front.security_id)]["last_price"]
            basis = fut - spot
            basis_win.append(basis)
            s = pd.Series(basis_win)
            z = ((basis - s.mean()) / s.std()) if len(s) >= 3 and s.std() else float("nan")

            net, ce_oi, pe_oi = net_option_delta(expiry)
            dnet = (net - prev_net) if prev_net is not None else 0.0
            prev_net = net
            pcr = (pe_oi / ce_oi) if ce_oi else float("nan")

            print(f"{now.strftime('%H:%M:%S')}  {spot:9.2f} {fut:9.2f} {basis:7.2f} "
                  f"{z:6.2f}  {net/1e6:11.2f} {dnet/1e6:+9.2f}  {pcr:5.2f}")
        except Exception as e:
            print(f"{now.strftime('%H:%M:%S')}  tick error: {e}")

        n += 1
        if args.iters and n >= args.iters:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
