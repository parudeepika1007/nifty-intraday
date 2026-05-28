"""Thin Dhan REST client. Only what we need for /v2/charts/intraday."""
from __future__ import annotations

import os
import time
from datetime import date

import requests
from dotenv import load_dotenv

load_dotenv()
DHAN_BASE = "https://api.dhan.co/v2"


def _headers() -> dict[str, str]:
    cid = os.environ.get("DHAN_CLIENT_ID")
    tok = os.environ.get("DHAN_ACCESS_TOKEN")
    if not cid or not tok:
        raise RuntimeError("DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN must be set in .env")
    return {
        "access-token": tok,
        "client-id": cid,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def fetch_intraday(
    security_id: int,
    exchange_segment: str,   # 'NSE_EQ' | 'NSE_FNO' | 'IDX_I'
    instrument: str,         # 'EQUITY' | 'FUTIDX' | 'OPTIDX' | 'INDEX'
    start: date,
    end: date,
    interval: str = "1",     # '1' | '5' | '15' | '25' | '60'
) -> dict:
    body = {
        "securityId": str(security_id),
        "exchangeSegment": exchange_segment,
        "instrument": instrument,
        "interval": interval,
        "fromDate": start.isoformat(),
        "toDate": end.isoformat(),
    }
    r = requests.post(
        f"{DHAN_BASE}/charts/intraday",
        headers=_headers(),
        json=body,
        timeout=45,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Dhan /charts/intraday {r.status_code}: {r.text[:300]}")
    time.sleep(0.3)
    return r.json()
