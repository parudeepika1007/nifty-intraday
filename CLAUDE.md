# NIFTY Intraday — Project Context

> **Read this first.** This is a handoff doc for a fresh Claude Code session
> in a new VS Code window. It captures *why* this repo exists, what's been
> decided, what to do next, and the exact code for the initial scaffold.

Last updated: 2026-05-29 (session in companion `Trading` repo).

---

## 1. Project goal

Discover a statistically valid **intraday directional edge for NIFTY** using
Dhan API data: spot + futures + options + OI + Greeks + IV + VIX + order
flow. Outcome we care about: predict the *short-term* direction of NIFTY
(next 15–30 minutes) with hit-rate measurably above 50%.

This is **research first, production second**. The first ~1 week is pure
backtesting against Dhan's historical 1-min data. We only build the live
WebSocket execution loop *after* a signal validates statistically.

User profile: Indian retail trader (Buddy) with Dhan broker access, F&O
positional + intraday research interest. The existing stockscanner repo
handles EOD positional; this project is intraday only.

---

## 2. Companion project — the EOD scanner

The user has another repo at `c:\Users\HP\Documents\PNR\Earn\Trading\` —
the production NSE F&O scanner. **Do not modify that repo from this one.**
It's a daily bhavcopy-driven, 7-scanner system (F&O daily, Swing, RS
Reversal, Price+OI Percentile, OI Strike-Breach, OI Inflection, Futures
Basis) running on GitHub Actions, writing to Supabase, served by a Next.js
dashboard on Vercel.

Why we kept these separate:
- **Vercel cannot host the intraday loop** — serverless can't maintain a
  6-hour WebSocket. Hard architectural blocker.
- **Different release cadence** — EOD changes nightly via cron; intraday
  research churns much faster.
- **Different storage pattern** — EOD = daily rows in Supabase; intraday =
  ~150k rows/day of 1-min bars, best in local DuckDB during research.
- **Different blast radius** — bugs in intraday research must not touch
  the production EOD scanner that the user is trading off.

Cross-references are OK (e.g. "the EOD Futures Basis scanner shipped in
`Trading/scanner/src/futures_basis.py` is the daily cousin of the
intraday `basis_regime.py` backtest here") — but never edit across.

---

## 3. Decisions already made (locked in)

| Question | Decision | Reason |
|---|---|---|
| Storage | **Local DuckDB** at `data/intraday.duckdb` | Columnar OLAP on 1-min bars, single-file backups, zero infra during research |
| Compute | **Laptop during market hours** for now | Free, easy to iterate; promote to Oracle Cloud Free Tier if a signal validates |
| Data source | **Dhan REST `/v2/charts/intraday`** | Has 1/5/15/25/60-min OHLCV + **OI** for futures & options, 5 years history |
| Data window | **2 years of 1-min** for NIFTY spot + last 6–8 monthly futures | Enough for statistical significance, fits free tier comfortably |
| Backtest harness | Standalone Python under `src/backtests/` | Same style as EOD repo's `scanner/research/edge_research.py` |
| Backtest output | **5-row z-quintile table** with n, hit%, median bps, mean bps | Identical format to EOD scanner sweeps; user can read at a glance |
| First idea to test | **Intraday basis regime** (Idea #2 from §4) | Strongest theoretical foundation, extends the EOD scanner |

These are not up for debate unless the user changes their mind.

---

## 4. The 5 candidate edges (ranked, build in this order)

### #2 (build first) — Intraday Basis Regime

**Market logic.** NIFTY futures normally carry a small positive basis (cost
of carry). Sharp intraday basis flips — especially through zero with
momentum — reveal institutional repositioning that hasn't fully reached
spot yet. Index futures are the cleanest read on institutional sentiment
because retail crowds into options, not index futures.

**Signal.** At each 1-min bar:
```
basis_t        = NIFTY_FUT_close - NIFTY_spot_close
basis_z        = (basis_t - 60-bar intraday mean) / 60-bar intraday stdev
basis_momentum = basis_t - basis_t-30
```
Bucket every minute into z-score quintile. Measure forward 15-min and
30-min NIFTY spot return per quintile. **Edge if Q5-Q1 spread > 2-3 bps**
with stat-significant t-stat.

**Failure conditions.** Expiry day (last Thursday) — index-arb dominates,
basis becomes mechanical. First 5 min of session — overnight reset noise.
Last 5 min — closing auction distorts.

### #1 — Opening Range Breakout + Basis Filter

**Market logic.** First 15 min absorbs overnight news. By 9:30 IST the
institutional bias is visible. Decades of evidence (Crabel et al.) shows
ORB has positive expectancy when filtered by *day type*. Basis confirms
direction.

**Signal.** Long when spot breaks above 9:15-9:30 high AND basis premium AND
volume > 1.2× 30-day median. Mirror for short.

**Failure conditions.** Range days (low ATR) — no break. Gap > 0.6% —
ORB fails, gap-fade works instead. Event days (RBI, Budget, election) —
macro overrides everything; pre-blacklist these dates.

### #5 — Time-of-Day Conditional Probability (a multiplier, not a signal)

P(afternoon trend continues | morning bucket) as a *position-sizing
adjustment* on #1 and #2. E.g. if morning bucket = "Trend Up" and that
historically continues into PM 62% of the time, size up longs and size
down counter-trend shorts.

### #3 — Option Greeks Net Positioning

Net delta of the whole option chain reveals institutional positioning.
Sharp shifts in net delta within a session = institutional repositioning.
Requires computing Black-Scholes Greeks from OI + IV at each minute
(~30 lines of Python; Newton-Raphson IV solve). Defer until #1/#2 prove
the infrastructure works.

### #4 — India VIX vs NIFTY Divergence

NIFTY and India VIX have correlation ~-0.78 normally. When they move
*together* (same sign), institutions are hedging — leading signal for
reversal within 1-3 hours. Smallest expected edge of the five; lowest
priority.

---

## 5. Dhan API specifics

### Endpoint we'll use

```
POST https://api.dhan.co/v2/charts/intraday

Headers:
  access-token: <DHAN_ACCESS_TOKEN>
  client-id: <DHAN_CLIENT_ID>
  Content-Type: application/json

Body:
  {
    "securityId": "13",                # numeric ID from Dhan master CSV
    "exchangeSegment": "IDX_I",         # NSE_EQ | NSE_FNO | IDX_I
    "instrument": "INDEX",              # EQUITY | FUTIDX | OPTIDX | INDEX
    "interval": "1",                    # 1 | 5 | 15 | 25 | 60 (min)
    "fromDate": "2026-04-01",
    "toDate": "2026-04-30"
  }

Response (column-parallel arrays):
  {
    "open":          [...],
    "high":          [...],
    "low":           [...],
    "close":         [...],
    "volume":        [...],
    "open_interest": [...],     # only for futures/options
    "timestamp":     [...]      # Unix seconds, UTC
  }
```

### Hard constraints

- **Max 90 days per call.** Caller paginates.
- Timestamps are UTC seconds → convert to `Asia/Kolkata` for IST display.
- `open_interest` is `null` / missing for cash equities and index spot.
- Rate limit on `/charts/intraday` is generous (multiple req/sec), but
  we sleep 0.3s between calls to stay polite.

### Known security_ids

- **NIFTY 50 spot**: `13` on segment `IDX_I`, instrument `INDEX`. Stable.
- **NIFTY monthly futures**: change every expiry. Look them up from
  Dhan's master CSV at `https://images.dhan.co/api-data/api-scrip-master.csv`.
  Filter rows where `SEM_EXM_EXCH_ID == "NSE"`,
  `SEM_INSTRUMENT_NAME == "FUTIDX"`,
  `SEM_TRADING_SYMBOL` starts with `NIFTY-`.

### Credentials

User has a personal Dhan account. Credentials go in `.env`:
```
DHAN_CLIENT_ID=<numeric client id>
DHAN_ACCESS_TOKEN=<JWT, rotates periodically>
```
Both `.env` and `data/*.duckdb` are gitignored.

---

## 6. Repo layout (the initial scaffold)

```
nifty-intraday/
├── CLAUDE.md                   ← you are here
├── README.md                   quick-start
├── .env.example                Dhan creds template
├── .gitignore                  ignore .env, data/, __pycache__
├── requirements.txt            duckdb, pandas, requests, python-dotenv
├── data/
│   └── .gitkeep                DuckDB file lives here (gitignored)
└── src/
    ├── __init__.py
    ├── dhan.py                 thin /v2/charts/intraday client
    ├── instruments.py          NIFTY spot + futures resolution
    ├── schema.py               DuckDB table + indexes
    ├── pull_historical.py      backfill driver (90-day chunks)
    └── backtests/
        ├── __init__.py
        └── basis_regime.py     first idea (see §4 #2)
```

The exact code for every file is in §10 (Appendix). On first session,
scaffold the repo from there. About 250 lines of Python total.

---

## 7. Backtest conventions (carried over from the EOD repo)

- **Walk-forward only.** Forward returns measured AT or AFTER the signal
  bar. Never peek at future data.
- **Reset rolling features at session boundaries.** Don't let yesterday's
  last basis carry into today's first 9:15:00 bar.
- **Aggregate by quintile/decile first, not by point estimate.** Single
  thresholds (e.g. "basis_z > 2") are too binary; quintile spreads are
  more honest.
- **Report n, hit%, median bps, mean bps, std bps.** Same shape as EOD
  scanner's `summarize()` helper. The user reads these numbers fast.
- **Always print a Q5-Q1 spread + hit-rate uplift line.** That's the
  one-line summary of "did the signal work."
- **Forward returns in basis points** (bps), not %. Intraday moves are
  small; 1 bp = 0.01%.
- **Filter event days** at the *report* level, not in the data: keep
  them in the DuckDB; flag and exclude in the backtest. List of event
  days lives in `src/events.py` (build when needed).

---

## 8. Success criterion for moving from research → live

A signal "validates" and gets promoted to live execution only if **all**:

1. **Q5-Q1 spread ≥ 3 bps** on 30-min forward returns
2. **t-stat of the spread > 2** (so ~95% confidence it's not noise)
3. **Hit-rate uplift Q5 vs Q1 ≥ 5 percentage points**
4. **At least 500 sample points** in each of Q1 and Q5
5. **Holds out-of-sample** — split the 2 years 80/20, fit features on
   the first 80%, test on the last 20%. The numbers above must be met
   on the OOS slice, not just in-sample.

If a signal meets all five, build the live Dhan WebSocket loop. If not,
the idea is dead — write a brief "rejected because…" doc and move to
the next idea on the ranked list in §4.

---

## 9. Next steps in order

### Phase A — Scaffold (first session, ~30 min)

1. `mkdir` the directories from §6.
2. Write each file from §10 Appendix verbatim.
3. `pip install -r requirements.txt`.
4. Copy `.env.example` to `.env` and ask the user to fill creds.
5. `python -m src.schema` — confirm `data/intraday.duckdb` exists.
6. `git init` and make the first commit.

### Phase B — Validate one round-trip (30 min)

7. Run `python -m src.pull_historical --years 0` and confirm a small
   pull returns rows (default `--years 0` should mean today only;
   adjust if needed).
8. Spot-check the DuckDB with `duckdb data/intraday.duckdb -c
   "SELECT COUNT(*), MIN(ts_minute), MAX(ts_minute) FROM bars_1min;"`

### Phase C — Backfill 2 years (~30 min wall-clock, mostly sleeping)

9. `python -m src.pull_historical --years 2`. Logs progress per chunk.
10. After the pull, confirm row count: should be ~150k rows for NIFTY
    spot alone (375 bars × 250 days × 2 yrs ≈ 187k, but holidays reduce
    it). Plus 60-100k per futures contract.

### Phase D — First backtest (30 min)

11. `python -m src.backtests.basis_regime`.
12. Read the printed table. If Q5-Q1 spread is meaningfully positive,
    iterate on parameter choices (z-window, forward horizon).
13. If totally flat, hypothesise *why* — likely the basis 1-min noise
    is too small to dominate spread; consider 5-min bars or longer
    z-window.

### Phase E — Iterate, then decide

14. Sweep variants the same way the EOD repo does (multiple parameter
    sets, compare side-by-side).
15. Apply the §8 success criteria.
16. Either: promote to live infra (new module under `src/live/`) OR
    move on to idea #1 (ORB).

---

## 10. Things that will go wrong

- **Dhan auth token rotates.** If `/charts/intraday` returns 401, the
  user needs to regenerate the access token from Dhan dashboard.
- **Master CSV column names occasionally change.** Dhan has done this
  twice in the past 18 months. If `list_nifty_futures()` returns empty,
  print the columns and re-map.
- **Timestamps tz quirks.** Dhan returns UTC seconds. We store IST
  naive timestamps (`Asia/Kolkata` localized → tz-stripped). DuckDB
  comparisons must use the same convention.
- **Holiday dates have NO bars.** The backfill silently returns empty
  responses for NSE holidays — fine, just don't panic if a date is
  missing.
- **Futures liquidity collapses in expiry week.** The "near-month"
  future on a Wednesday before expiry is mechanically pinned to spot
  by arb activity. Basis signals on those days are noise — flag and
  exclude.
- **Gap days break the rolling z-score.** The first 30-60 minutes of
  a session with a >0.5% gap from yesterday close will produce
  spuriously large z-scores. Either skip the first hour or use a
  longer warmup window.

---

## 11. Memory / collaboration notes

These come from the user's persistent memory and apply to ALL their
projects:

- **No third-party product names in UI or code comments.** Never
  reference Quantsapp, TradingView, Zerodha, etc. in any rendered
  string or comment. (TradingView Lightweight Charts as a library
  *dependency* is fine; mentioning "TradingView" in user-visible UI is
  not.)
- **No Co-Authored-By trailer in commits.** Plain commit messages, no
  "Co-Authored-By: Claude" line. The user has this in `git log` already
  for the companion repo and wants it consistent.
- **All times in IST when communicating with the user.** Internal
  storage stays UTC; print/log in IST.
- **Backtest first, ship second.** The user trusts backtested numbers,
  not hand-waved "this should work" pitches. Same standard as the EOD
  repo's OI Inflection / Futures Basis decisions.

---

## 12. References

- Dhan historical-data docs: https://dhanhq.co/docs/v2/historical-data/
- Dhan master CSV: https://images.dhan.co/api-data/api-scrip-master.csv
- Companion EOD scanner: `c:\Users\HP\Documents\PNR\Earn\Trading\`
  (especially `scanner/research/edge_research.py` for the backtest
  harness style we're carrying over).

---

## 13. Appendix — full code for the initial scaffold

Create these files verbatim. Total ~250 lines.

### `.env.example`

```
DHAN_CLIENT_ID=
DHAN_ACCESS_TOKEN=
```

### `.gitignore`

```
.env
data/*.duckdb
data/*.parquet
__pycache__/
*.pyc
.venv/
.vscode/
```

### `requirements.txt`

```
duckdb>=1.0.0
pandas>=2.1.0
requests>=2.31.0
python-dotenv>=1.0.0
```

### `src/__init__.py` and `src/backtests/__init__.py`

Both empty files.

### `src/schema.py`

```python
"""DuckDB schema for intraday 1-min bars.

One table for all instruments (NIFTY spot, futures, options). The
`instrument_kind` discriminator makes per-type queries cheap, and the
composite PK (security_id, ts_minute) is the natural dedup key on
re-pulls.
"""
import duckdb
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "intraday.duckdb"


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def init_db() -> None:
    con = connect()
    con.execute("""
        CREATE TABLE IF NOT EXISTS bars_1min (
          security_id     BIGINT       NOT NULL,
          symbol          VARCHAR      NOT NULL,
          instrument_kind VARCHAR      NOT NULL,
          expiry          DATE,
          strike          DECIMAL(12,2),
          ts_minute       TIMESTAMP    NOT NULL,
          open            DECIMAL(14,4),
          high            DECIMAL(14,4),
          low             DECIMAL(14,4),
          close           DECIMAL(14,4),
          volume          BIGINT,
          oi              BIGINT,
          PRIMARY KEY (security_id, ts_minute)
        );
        CREATE INDEX IF NOT EXISTS idx_bars_ts ON bars_1min(ts_minute);
        CREATE INDEX IF NOT EXISTS idx_bars_sym ON bars_1min(symbol, ts_minute);
    """)
    con.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialised {DB_PATH}")
```

### `src/dhan.py`

```python
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
```

### `src/instruments.py`

```python
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
```

### `src/pull_historical.py`

```python
"""Pull N years of 1-min NIFTY spot + monthly futures into DuckDB.

Usage:
    python -m src.pull_historical
    python -m src.pull_historical --years 1
"""
from __future__ import annotations

import argparse
import datetime as dt
from datetime import date

import pandas as pd

from .dhan import fetch_intraday
from .instruments import NIFTY_SPOT, Future, list_nifty_futures
from .schema import connect, init_db

DHAN_MAX_DAYS = 90


def _flatten(raw: dict, *, security_id: int, symbol: str, kind: str,
             expiry: date | None, strike: float | None) -> pd.DataFrame:
    if not raw or "timestamp" not in raw or not raw["timestamp"]:
        return pd.DataFrame()
    return pd.DataFrame({
        "security_id": security_id,
        "symbol": symbol,
        "instrument_kind": kind,
        "expiry": expiry,
        "strike": strike,
        "ts_minute": pd.to_datetime(raw["timestamp"], unit="s", utc=True
                                    ).tz_convert("Asia/Kolkata").tz_localize(None),
        "open": raw["open"],
        "high": raw["high"],
        "low": raw["low"],
        "close": raw["close"],
        "volume": raw["volume"],
        "oi": raw.get("open_interest", [None] * len(raw["timestamp"])),
    })


def backfill_instrument(
    con, *, security_id: int, exchange_segment: str, instrument: str,
    symbol: str, kind: str, start: date, end: date,
    expiry: date | None = None, strike: float | None = None,
) -> int:
    total = 0
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + dt.timedelta(days=DHAN_MAX_DAYS - 1), end)
        print(f"  {symbol}  {chunk_start} → {chunk_end}", end="  ")
        raw = fetch_intraday(security_id, exchange_segment, instrument,
                             chunk_start, chunk_end)
        df = _flatten(raw, security_id=security_id, symbol=symbol, kind=kind,
                      expiry=expiry, strike=strike)
        if not df.empty:
            con.register("df_in", df)
            con.execute("INSERT OR REPLACE INTO bars_1min SELECT * FROM df_in")
            con.unregister("df_in")
        print(f"+{len(df)} rows")
        total += len(df)
        chunk_start = chunk_end + dt.timedelta(days=1)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2)
    args = ap.parse_args()

    init_db()
    con = connect()

    end = date.today()
    start = end - dt.timedelta(days=args.years * 365)
    print(f"Backfilling NIFTY 1-min bars  {start} → {end}")

    backfill_instrument(con, **NIFTY_SPOT, kind="spot", start=start, end=end)

    recent = [f for f in list_nifty_futures()
              if f.expiry >= start - dt.timedelta(days=30)][:8]
    for f in recent:
        backfill_instrument(
            con, security_id=f.security_id,
            exchange_segment="NSE_FNO", instrument="FUTIDX",
            symbol=f.symbol, kind="fut", expiry=f.expiry,
            start=max(start, f.expiry - dt.timedelta(days=60)),
            end=min(end, f.expiry),
        )
    con.close()
    print("Done.")


if __name__ == "__main__":
    main()
```

### `src/backtests/basis_regime.py`

```python
"""Backtest: NIFTY intraday basis regime.

Hypothesis: basis = (NIFTY_FUT - NIFTY_spot) carries directional info
at the 1-min level. Extreme basis z-scores predict short-term forward
spot direction.

Reports: 5-row quintile table with n, hit%, median bps, mean bps.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from ..schema import DB_PATH


def load_aligned(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = """
    WITH front AS (
        SELECT ts_minute, close, expiry, symbol,
               ROW_NUMBER() OVER (PARTITION BY ts_minute ORDER BY expiry ASC) AS rn
        FROM bars_1min
        WHERE instrument_kind = 'fut' AND expiry >= ts_minute::DATE
    )
    SELECT s.ts_minute, s.close AS spot, f.close AS fut, f.symbol AS fut_sym
    FROM bars_1min s
    JOIN front f ON f.ts_minute = s.ts_minute AND f.rn = 1
    WHERE s.symbol = 'NIFTY_SPOT'
    ORDER BY s.ts_minute
    """
    return con.sql(sql).df()


def compute_basis_signals(df: pd.DataFrame, zwin: int = 60) -> pd.DataFrame:
    df = df.copy()
    df["basis"] = df["fut"] - df["spot"]
    df["basis_pct"] = df["basis"] / df["spot"] * 100.0
    df["trade_date"] = df["ts_minute"].dt.date

    def zscore_intraday(g: pd.DataFrame) -> pd.DataFrame:
        b = g["basis"]
        roll_mean = b.rolling(zwin, min_periods=zwin // 2).mean()
        roll_std = b.rolling(zwin, min_periods=zwin // 2).std()
        g["basis_z"] = (b - roll_mean) / roll_std
        return g

    df = df.groupby("trade_date", group_keys=False).apply(zscore_intraday)
    df["fwd_15m_bps"] = (df["spot"].shift(-15) / df["spot"] - 1) * 10000
    df["fwd_30m_bps"] = (df["spot"].shift(-30) / df["spot"] - 1) * 10000
    return df


def report(df: pd.DataFrame) -> None:
    df = df.dropna(subset=["basis_z", "fwd_30m_bps"]).copy()
    df["z_bin"] = pd.qcut(df["basis_z"], q=5,
                          labels=["Q1_low", "Q2", "Q3_mid", "Q4", "Q5_high"])

    rows = []
    for label, grp in df.groupby("z_bin", observed=True):
        rows.append({
            "z_bin": label,
            "n": len(grp),
            "z_median": grp["basis_z"].median(),
            "hit_15m": (grp["fwd_15m_bps"] > 0).mean() * 100,
            "mean_15m_bps": grp["fwd_15m_bps"].mean(),
            "hit_30m": (grp["fwd_30m_bps"] > 0).mean() * 100,
            "mean_30m_bps": grp["fwd_30m_bps"].mean(),
        })
    out = pd.DataFrame(rows)
    print("\nBasis Regime backtest — forward returns by basis z-quintile")
    print("=" * 78)
    print(out.to_string(index=False, float_format="%.2f"))

    top = out.iloc[-1]
    bot = out.iloc[0]
    print(f"\nQ5 - Q1 spread:  15m {top['mean_15m_bps'] - bot['mean_15m_bps']:+.2f} bps  "
          f"30m {top['mean_30m_bps'] - bot['mean_30m_bps']:+.2f} bps")
    print(f"Q5 hit-rate uplift vs Q1:  15m {top['hit_15m'] - bot['hit_15m']:+.1f}pp  "
          f"30m {top['hit_30m'] - bot['hit_30m']:+.1f}pp")


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    raw = load_aligned(con)
    print(f"Loaded {len(raw):,} aligned 1-min bars  "
          f"({raw['ts_minute'].min()} → {raw['ts_minute'].max()})")
    df = compute_basis_signals(raw)
    report(df)
    con.close()


if __name__ == "__main__":
    main()
```

### `README.md`

```markdown
# NIFTY Intraday — research repo

Backtest 1-min intraday NIFTY signals using Dhan historical data.

## Setup
    python -m venv .venv && .venv\Scripts\activate
    pip install -r requirements.txt
    copy .env.example .env       # fill DHAN_CLIENT_ID + DHAN_ACCESS_TOKEN
    python -m src.schema

## Pull 2 years of NIFTY data
    python -m src.pull_historical

## Run the first backtest
    python -m src.backtests.basis_regime
```

---

## 14. First message in a fresh session

When a new Claude Code session opens this repo, the user's first message
will likely be:

- "Set up the project" → scaffold all files from §13.
- "Run the backfill" → assume scaffolding is done, the user has filled
  `.env`, and proceed with `python -m src.pull_historical`.
- "Run the backtest" → assume backfill is done, proceed with
  `python -m src.backtests.basis_regime`, report results, propose
  next experiment.
- "Add idea #N" → look up the idea in §4, draft a new file under
  `src/backtests/`, write methodology section, then code.

In all cases: **don't second-guess decisions already locked in §3**.
If a decision needs revisiting, surface it explicitly and wait for
confirmation.
