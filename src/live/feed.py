"""Live Dhan market-data client — READ ONLY (snapshots, no order placement).

Wraps the three live endpoints we need for the #2 (basis) and #3 (option
greeks) live monitor:

  * /v2/marketfeed/ltp           last price for many securities at once
  * /v2/optionchain/expirylist   available expiries for an underlying
  * /v2/optionchain              per-strike CE/PE with greeks + OI + IV

Rate limits (per Dhan docs): marketfeed = 1 req/s, optionchain = 1 req/3s.
We sleep accordingly so the caller can poll in a tight loop politely.
"""
from __future__ import annotations

import time

import requests

from ..dhan import DHAN_BASE, _headers


def _post(path: str, body: dict, *, sleep: float, timeout: int = 30) -> dict:
    r = requests.post(f"{DHAN_BASE}/{path}", headers=_headers(), json=body,
                      timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Dhan /{path} {r.status_code}: {r.text[:300]}")
    time.sleep(sleep)
    return r.json()


def marketfeed_ltp(segments: dict[str, list[int]]) -> dict:
    """segments e.g. {"IDX_I": [13], "NSE_FNO": [62329]} -> last prices."""
    return _post("marketfeed/ltp", segments, sleep=1.0, timeout=20)["data"]


def option_expiry_list(scrip: int = 13, seg: str = "IDX_I") -> list[str]:
    body = {"UnderlyingScrip": scrip, "UnderlyingSeg": seg}
    return _post("optionchain/expirylist", body, sleep=3.0, timeout=20)["data"]


def option_chain(expiry: str, scrip: int = 13, seg: str = "IDX_I") -> dict:
    """Returns {"last_price": float, "oc": {strike: {"ce":..., "pe":...}}}."""
    body = {"UnderlyingScrip": scrip, "UnderlyingSeg": seg, "Expiry": expiry}
    return _post("optionchain", body, sleep=3.0, timeout=30)["data"]
