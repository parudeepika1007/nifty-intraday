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
