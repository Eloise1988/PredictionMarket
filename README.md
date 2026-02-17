# Prediction Market -> Equity Signal Agent

Decision-support agent that ingests prediction market data, filters high-liquidity/high-conviction contracts, uses an LLM to discover impacted stocks per market dynamically, ranks by valuation fit, and sends Telegram alerts.

## What this is
- Real-time monitoring and ranking of *potentially impacted* stocks.
- Risk-aware signal generation, not auto-trading.
- Built for Ubuntu 20 + MongoDB + OpenAI API + Telegram Bot API.

## Core flow
1. Fetch open markets (default: Polymarket-wide scan).
2. Keep only markets above liquidity threshold and with extreme probabilities (default <=30% or >=70%).
3. For each eligible market, query LLM for impacted U.S. stocks/ETFs and direction if event resolves YES.
4. Pull valuation/quote data for discovered tickers.
5. Select the best-valuation ticker per market, score and rank ideas.
6. Deduplicate alerts with MongoDB cooldown digest.
7. Send Telegram message with rationale and ticker background.

## Iteration 2 features
- Websocket ingestion service for lower-latency updates:
```bash
python3 -m prediction_agent.streaming.service --dry-run
```
- Backtest + walk-forward threshold calibration:
```bash
python3 -m prediction_agent.backtest.runner --save-report
```
- Backtest window example:
```bash
python3 -m prediction_agent.backtest.runner --start 2025-01-01 --end 2025-12-31 --save-report
```
- Built-in unit tests:
```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
- Company/ETF background lines are included in alert output and LLM-ranked ideas for clearer rationale.

## Dynamic mapping knobs
- `POLYMARKET_ONLY_MODE=true`
- `FINANCE_ONLY_MODE=true`
- `TOP_LIQUIDITY_FINANCE_MARKETS=5`
- `MIN_SIGNAL_LIQUIDITY=100000`
- `PROBABILITY_LOW_THRESHOLD=0.30`
- `PROBABILITY_HIGH_THRESHOLD=0.70`
- `MAX_MARKETS_FOR_LLM=5`
- `LLM_MAP_MAX_TICKERS=8`
- No fixed ticker universe is required; tickers are discovered per-market by the LLM.
- Selection logic: scan finance markets by liquidity and return the first 5 that pass liquidity/probability filters.

These are set in `.env` and can be tuned without code changes.

## Quick start (local)
1. Create environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Configure env:
```bash
cp .env.example .env
# Fill API keys and Telegram chat settings
```
3. Start MongoDB (local or docker):
```bash
docker compose up -d mongodb
```
4. Run one cycle safely:
```bash
python3 -m prediction_agent.app --once --dry-run
```
5. Run continuous polling:
```bash
python3 -m prediction_agent.app
```

Or use Make targets:
```bash
make run-once
make stream-dry-run
make backtest
make test
```

## Docker
```bash
cp .env.example .env
# Fill .env first

docker compose up --build
```

## Ubuntu 20 systemd deployment
1. Clone repo to `/opt/prediction-agent`.
2. Create virtualenv and install dependencies.
3. Copy `deploy/prediction-agent.service` to `/etc/systemd/system/`.
4. Update `User`, `Group`, and paths in the unit file if needed.
5. Start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now prediction-agent
sudo systemctl status prediction-agent
```

## Signal model
Final score is a weighted blend of:
- probability edge from prediction market (distance from 50%)
- market quality (liquidity + volume + freshness)
- event->equity exposure weight
- valuation attractiveness (PE/PB/EV-EBITDA)
- momentum consistency

## Real-time best practices included
- provider abstraction and retry logic
- MongoDB TTL indexes for storage hygiene
- idempotent digest-based alert suppression
- configurable cooldown windows
- minimum edge/liquidity filters to reduce low-quality contracts
- cross-market confirmation bonus to reduce single-venue noise
- keyword boundary matching + sports-market filtering to reduce false thematic matches
- dry-run mode for safe rollout

## Backtest bias controls included
- next-day trade entry to avoid look-ahead bias
- chronological walk-forward calibration (train then validate forward in time)
- transaction-cost deduction per trade
- minimum trades requirement before accepting calibrated thresholds
- report persistence in Mongo for audit trail
- optional bias diagnostics report (`sample size`, `threshold stability`, `theme concentration`, `drawdown`)

Detailed protocol: `docs/BIAS_CONTROLS.md`.

## Important risk notes
- This is not investment advice.
- Prediction markets can be noisy and manipulated in low-liquidity contracts.
- Alpha Vantage free tier is not enough for high-frequency production; use a higher-tier market data provider for live deployment.
- Add hard risk controls before any capital deployment (position sizing, max drawdown stop, theme concentration limits, and trade execution safeguards).
# PredictionMarket
