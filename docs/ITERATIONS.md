# Demand + Feasibility + Iteration Plan

## Demand evaluation
- **Who wants this:** discretionary macro traders, event-driven hedge funds, geopolitical/news desks, and advanced retail traders.
- **Why demand exists:** prediction markets react quickly to information shocks and can precede analyst revisions in sector-specific names.
- **Practical willingness to use:** high for alerting/research workflows; lower for fully automated execution unless false-positive control is strong.

## Feasibility evaluation
- **Technical feasibility:** high for alerting and ranking, medium for robust alpha generation, low for unattended auto-trading without a larger risk stack.
- **Data feasibility:** high for prediction market inputs, medium for institutional-grade equity/fundamental data at low cost.
- **Operational feasibility:** high with polling; medium-high with websocket/event streaming and queueing.

## Iteration 1 (implemented)
- Polymarket + Kalshi ingestion
- Rule-based event/theme extraction
- Event->equity exposure map
- Multi-factor score (probability edge + quality + valuation + momentum)
- Cross-market consensus aggregation (source confirmation bonus)
- MongoDB persistence and alert dedupe
- Telegram push alerts + OpenAI explanation layer

## Iteration 2 (implemented baseline)
- Websocket ingestion service for lower latency (Kalshi ticker channel + Polymarket CLOB market stream)
- Batch processing from stream into the same scoring/risk engine
- Backtesting harness with next-day entry and transaction-cost-adjusted returns
- Walk-forward threshold calibration on historical idea snapshots
- Unit tests for backtest engine behavior

## Iteration 2.5 (recommended next)
- Add market microstructure checks (spread, depth, stale quote detection)
- Add Bayesian probability smoothing across venues
- Add feature store and model registry for reproducible research

## Iteration 3 (institutional hardening)
- Replace low-tier pricing feeds with institutional data APIs
- Add portfolio construction constraints (sector caps, beta-neutral overlays)
- Add execution simulator and transaction-cost model
- Add anomaly detection for manipulation / spoofing behavior in thin markets
- Deploy with queue + worker architecture (Kafka/Redis streams) and full observability (Prometheus/Grafana + on-call alerts)

## Performance improvements with highest ROI
1. **Better equity data quality** (real-time quotes + fundamentals from a stronger feed).
2. **Cross-market confirmation** (same thesis confirmed by multiple prediction venues).
3. **Backtest + calibration** (optimize threshold by regime, not fixed constants).
4. **Risk engine before execution** (max loss/day, per-theme caps, portfolio correlation limits).
