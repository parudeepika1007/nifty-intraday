"""NIFTY spot + monthly-futures security_id resolution from Dhan master CSV.

Spot is stable (NIFTY 50 = 13 on IDX_I). Futures change every month;
we look them up at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO

import pandas as pd
import requests

SECURITY_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"


NIFTY_SPOT = {
    "security_id": 13,
    "exchange_segment": "IDX_I",
    "instrument": "INDEX",
    "symbol": "NIFTY_SPOT",
}


@dataclass(frozen=True)
class Future:
    security_id: int
    symbol: str           # e.g. "NIFTY-May2026-FUT"
    expiry: date


def fetch_master() -> pd.DataFrame:
    r = requests.get(SECURITY_MASTER_URL, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def list_nifty_futures(master: pd.DataFrame | None = None) -> list[Future]:
    df = master if master is not None else fetch_master()
    nifty_fut = df[
        (df["SEM_EXM_EXCH_ID"] == "NSE")
        & (df["SEM_INSTRUMENT_NAME"] == "FUTIDX")
        & (df["SEM_TRADING_SYMBOL"].str.startswith("NIFTY-"))
    ].copy()
    nifty_fut["expiry"] = pd.to_datetime(nifty_fut["SEM_EXPIRY_DATE"]).dt.date
    nifty_fut = nifty_fut.sort_values("expiry")
    return [
        Future(int(r.SEM_SMST_SECURITY_ID), r.SEM_TRADING_SYMBOL, r.expiry)
        for r in nifty_fut.itertuples(index=False)
    ]
