# NIFTY Intraday — Strategy Research Memo

> Written as a senior-quant research plan. The organising principle is not
> "which signal sounds smart" but **"which signal can be statistically
> validated with the data we can actually obtain."** A signal you cannot
> backtest is not an edge — it is a hypothesis with a story attached.

Last updated: 2026-05-29.

---

## 0. The binding constraint — data availability (read this first)

Everything below is gated by what Dhan actually lets us reconstruct. We
learned this the hard way this week:

| Data source | Historical (backtestable) | Live only (forward capture) | Why |
|---|---|---|---|
| NIFTY spot OHLC, 1-min, 2 yr | ✅ | | Stable index security_id (13) |
| India VIX OHLC, 1-min, 2 yr | ✅ | | Stable index security_id |
| NIFTY **futures** intraday | ❌ | ✅ | Expired contracts' Dhan IDs vanish from the live master; only the live front-month is reachable |
| **Option chain** (greeks, IV, OI) | ❌ | ✅ | `/optionchain` is a *live snapshot* only; expired option IDs are gone too |
| **OI changes** (fut + options) | ❌ | ✅ | Same expired-ID problem |
| **Order flow / 20-level depth** | ❌ | ✅ | Only via the live WebSocket feed; never stored historically |
| Spot **volume / VWAP** | ❌ | (futures vol live) | NIFTY index 1-min volume is 0 on ~72% of bars — unusable; true VWAP needs futures/option volume |

**Consequence for sequencing:**

- **Tier 1** (spot + VIX) can be **validated this week** on 2 years of data.
  These are the only ideas that can reach the §8 promote bar soon.
- **Tier 2** (futures basis, OI structure, greeks/GEX, order flow) is where the
  real institutional edge lives — but it has **no history**. The only honest
  path is to **record live snapshots forward into DuckDB now**, then backtest
  that captured panel in a few weeks. There is no shortcut.

This is exactly why the current live monitor "makes no signals": see §1.

---

## 1. Why the live monitor is silent — and the fix

The monitor uses **hand-set thresholds** (basis-z zero-cross + Δnet-delta > 1.5 mn)
joined by an **AND**. Three problems, all fixable, none requiring new data:

1. **Uncalibrated thresholds.** ±1.5 mn and "z crosses 0" are guesses. A quant
   never guesses a threshold — you set it at an empirical quantile (e.g. the
   top/bottom decile of *that session's own* Δ distribution).
2. **Rare trigger.** Requiring a zero-*cross* AND a same-tick delta spike is a
   near-empty intersection intraday. Real signals use **state** (z is in its
   top quintile) not **events** (z crossed exactly now).
3. **Binary verdict.** "SIGNAL / NO SIGNAL" throws away information. Replace
   with a **continuous probability** (0–100% bullish) so the user sees the
   read strengthening *before* it crosses any line.

**Fix:** drive the verdict off rolling **percentile ranks** computed from the
session's own distribution, blend the components into a logistic score, and
display a probability gauge. This is folded into the framework in §8.

---

## TIER 1 — validate now on 2 yr spot + VIX

### S1. Trend-day classification → conditional breakout

**1. Market logic.** Intraday edges are regime-dependent: breakouts pay on
*trend days* and bleed on *range days*. Decades of evidence (and our own ORB
test — wide opening ranges hit 58.6% at 30 min vs 46% for narrow) say the
opening 30–60 min *fingerprints* the day type. So don't trade breakouts blindly
— first **classify the day**, then only take breakouts on days predicted to trend.

**2. Measurable features** (all from spot OHLC, reset daily):
- Opening-range (09:15–09:45) width in bps, percentile-ranked vs trailing 60 days.
- First-hour realized volatility (std of 1-min returns) and its percentile.
- Open location vs prior-day range (gap %, inside/outside prior range).
- India VIX level percentile at 09:30 (high VIX → trend-prone). *(uses S2 data)*
- Number of OR-high/OR-low touches in the first hour (chop proxy).

**3. Backtesting methodology.** Label each historical day *trend* vs *range* by
its realised outcome (e.g. |close−open| / day-range > 0.6 = trend). Fit a
**logistic regression / gradient-boosted classifier** on the 09:45 features to
predict P(trend day). Walk-forward, 80/20 OOS. Then simulate: take the breakout
only when P(trend) > τ; measure forward 30-min and close returns by P(trend)
decile. Edge = monotonic lift in breakout win-rate across deciles.

**4. Probability / confidence.** The classifier *is* the confidence score —
output calibrated P(trend) (check with a reliability/Brier plot). Position size
∝ (P − 0.5).

**5. Failure conditions.** Event days (RBI/Budget/election) — pre-blacklist.
Expiry-day pinning. Regime shifts in volatility (refit rolling). Gap > 0.6%
flips the playbook to gap-fade.

**6. Python / Dhan.** Spot already in DuckDB. Build
`src/backtests/trend_day.py`: feature-build per day → `sklearn` LogisticRegression
(add to requirements) → quintile report in the house format. No new data pull.

---

### S2. VIX–NIFTY co-movement reversal *(the now-backtestable idea #4)*

**1. Market logic.** NIFTY and India VIX are structurally anti-correlated
(~−0.8): price up, fear down. When they break that and move the **same sign**
intraday (both up, or both down), it signals **institutional hedging /
repositioning** — option demand rising into a rally, or fear collapsing into a
sell-off — which historically precedes a **reversal within 1–3 hours**. Crucially,
both are indices → **fully backtestable on 2 yr of 1-min data.**

**2. Measurable features:**
- Rolling 30-min correlation of NIFTY vs VIX 1-min returns; flag when it turns
  **positive** (the regime break).
- Same-sign co-move intensity: `sign(ret_nifty)==sign(ret_vix)` streak length.
- VIX 1-min return z-score (spike detection).
- VIX level percentile (regime: <12 complacent, >18 stressed).
- Divergence magnitude = standardized (Δlog NIFTY + Δlog VIX).

**3. Backtesting methodology.** Pull India VIX 1-min for 2 yr (same loader as
spot). Align on `ts_minute`. Bucket every minute by co-move state (anti / neutral
/ same-sign) and by VIX-regime; measure forward 30/60/90-min NIFTY returns and
**reversal hit-rate** (does price reverse the prior 30-min move?). Quintile the
divergence magnitude. Walk-forward, OOS. Edge = same-sign bucket shows
significant mean-reversion vs anti-sign baseline.

**4. Probability / confidence.** Logistic P(reversal) on the features; confidence
scales with divergence magnitude and VIX-spike z.

**5. Failure conditions.** Trending macro days where VIX and NIFTY *legitimately*
fall together for hours (melt-up) — the signal mistimes. Expiry IV crush distorts
VIX. First/last 15 min. Low-VIX regimes mute the effect.

**6. Python / Dhan.** Add India VIX to `instruments.py` (resolve from master CSV
where symbol contains "VIX", segment IDX_I), extend `pull_historical.py` to
backfill it (2 indices, ~1 hr wall-clock), then `src/backtests/vix_divergence.py`.
Self-contained, no futures/options needed.

---

### S3. Time-of-day conditional drift *(idea #5, as a standalone test)*

**1. Market logic.** NIFTY's intraday session has structure: the morning sets a
bias, lunch (12:00–13:30) drifts, and the last hour trends or reverts depending
on the morning. P(afternoon continues | morning regime) is a stable, exploitable
conditional probability — and a natural **position-size multiplier** on S1/S2.

**2. Measurable features:** morning return bucket (09:15–11:00 signed move,
quintiled); morning range vs ATR; "first-hour high/low broken by 11:00?"; day of
week; days-to-expiry.

**3. Backtesting methodology.** Build the conditional probability table
P(13:30→15:30 return sign | morning bucket) over 2 yr, with confidence intervals
and OOS check. Report as the house quintile table.

**4. Probability / confidence.** The empirical conditional probability *is* the
score; attach Wilson confidence intervals (so thin buckets are honestly flagged).

**5. Failure conditions.** Event days; expiry; regime drift (recompute rolling).
Thin buckets → wide CIs → don't size up.

**6. Python / Dhan.** Pure spot, in DuckDB already. `src/backtests/tod_drift.py`.

---

## TIER 2 — institutional edges (no history; capture live, then validate)

These are where a hedge fund actually makes intraday money. None can be
backtested today — they require the **tick recorder** (§7) running for weeks
first. Designed now so capture records the right fields.

### S4. OI structure — option writers vs buyers

**1. Market logic.** Index option writers are institutions; they are right more
often than buyers. *Where* OI is being **added** maps the battlefield: heavy call
writing at a strike = ceiling; heavy put writing = floor; **unwinding** of those
= the level is about to break. Futures price+OI quadrants (long buildup / short
buildup / short covering / long unwinding) read institutional conviction.

**2. Features:** per-strike ΔOI (CE/PE) each minute; shift in max-CE-OI and
max-PE-OI strikes; futures Δprice×ΔOI quadrant; max-pain drift; ATM straddle
price (vol proxy) decay rate.

**3. Backtest:** forward only — record the chain each minute, then test whether
"put-writing buildup at/above spot" precedes up-moves, etc., via the quintile
framework. **4. Confidence:** logistic on quadrant + ΔOI magnitude.
**5. Failure:** expiry-day OI noise; illiquid far strikes; data gaps.
**6. Dhan:** `/optionchain` (greeks+OI per strike) + `/marketfeed/quote` (futures OI)
each minute → DuckDB.

### S5. Dealer gamma exposure (GEX) regime

**1. Logic.** Dealers hedge their option books; their **net gamma** dictates
whether they buy dips/sell rips (positive GEX → range/mean-revert day) or chase
(negative GEX → trend/volatile day). The **zero-gamma flip level** is a magnet/
trapdoor for spot. This single regime variable classifies trend-vs-range *and*
gives intraday support/resistance.

**2. Features:** `GEX(strike)=Σ γ·OI·spot²·0.01` summed CE−PE; total GEX sign;
zero-gamma price vs spot; distance to it. **3. Backtest:** forward — does
positive-GEX predict mean-reversion to VWAP, negative-GEX predict breakout
follow-through? **4. Confidence:** magnitude of |GEX| and distance to flip.
**5. Failure:** dealer-positioning sign is an assumption (we infer from OI, don't
see dealer books); expiry gamma explosion. **6. Dhan:** greeks come straight from
`/optionchain` — no Black-Scholes needed; aggregate per minute.

### S6. Order-flow / depth imbalance

**1. Logic.** Highest-frequency read: aggressive buy vs sell volume and bid/ask
depth imbalance reveal who's in control *right now*; persistent imbalance against
a flat tape = absorption → reversal. **2. Features:** top-5/20-level depth
imbalance, trade-side classification (tick rule), cumulative volume delta,
absorption events. **3. Backtest:** forward, sub-minute, via WebSocket capture.
**4. Confidence:** imbalance persistence/Z. **5. Failure:** spoofing, iceberg
orders, latency. **6. Dhan:** live **WebSocket** feed (Level-2 depth + tick) →
this is the most infra-heavy; defer until S4/S5 prove the capture pipeline.

---

## 8. Shared probability & confidence framework

Apply uniformly so signals are comparable and explainable:

1. **Feature → percentile.** Rank each feature against its *own session* (or
   trailing-N-day) distribution. Removes hand-set thresholds entirely.
2. **Blend → logistic.** `P(up) = σ(Σ wᵢ·featureᵢ)`; weights fit by
   walk-forward logistic regression, not chosen by hand.
3. **Calibrate.** Reliability curve + Brier score — a "70%" must mean 70%.
4. **Confidence band.** Wilson CI on every bucket; thin buckets flagged, never
   sized up.
5. **Validation gate (§8 of CLAUDE.md).** Q5−Q1 ≥ 3 bps @30 min, t > 2, hit
   uplift ≥ 5 pp, ≥ 500 pts/bucket, **holds OOS** (80/20). No pass → idea dies,
   write the rejection note, move on.
6. **Display.** The dashboard verdict becomes a **0–100 probability gauge**, not
   a binary — you watch conviction build in real time.

---

## 9. Recommended sequencing

1. **This week (validatable):** backfill India VIX → build **S2** (VIX
   divergence) and **S1** (trend-day classification). Both run on indices we have
   or can pull in an hour. These are the only ideas that can earn real money with
   *proven* numbers soon.
2. **In parallel, start the tick recorder** (§7 / S4 fields): persist
   `/optionchain` + futures quote every minute to DuckDB. This is the seed corn
   for S4/S5 — the genuinely institutional edges — which become backtestable only
   after weeks of capture.
3. **Then S5 (GEX), then S6 (order flow)** once the pipeline is proven.
4. Re-skin the live dashboard to the **probability gauge** (§1 fix) so it stops
   being silent and starts showing graded conviction.

**Bottom line:** the institutional edges you listed (OI, greeks, order flow) are
real and worth building — but they are **forward-capture** projects, not
backtests, because Dhan keeps no history for expired derivatives. The fastest
route to a *statistically valid, tradeable* edge right now is **VIX divergence +
trend-day classification on the 2-year index data we already control**, while the
recorder quietly accumulates the data the bigger edges need.
