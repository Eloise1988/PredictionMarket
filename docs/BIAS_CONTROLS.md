# Bias Controls and Validation Protocol

## Objective
Reduce false confidence from common financial backtest failures and prevent overfitting to noise.

## Bias controls implemented
- **Look-ahead control:** backtest enters at next trading day close after signal timestamp.
- **Chronological validation:** walk-forward training/validation splits only move forward in time.
- **Transaction cost modeling:** round-trip costs deducted from each trade.
- **Liquidity/edge gating:** live system drops weak-probability and low-liquidity contracts.
- **Threshold robustness:** score threshold selected from training windows and evaluated out-of-sample.

## Biases still requiring extra safeguards
- **Survivorship bias:** current ticker universe may miss delisted names.
- **Data snooping:** too many parameter sweeps can overfit; keep search space small.
- **Regime bias:** a strategy can fail when macro regime changes.
- **Execution bias:** alerts are not guaranteed executable at modeled prices.

## Operating checklist before capital deployment
1. Run backtest with at least 12 months of recorded ideas.
2. Use a strict out-of-sample holdout window never used for calibration.
3. Stress test by increasing transaction costs and slippage assumptions.
4. Cap position sizes and per-theme exposures.
5. Compare against simple baselines (SPY, sector ETF, random-timing control).
6. Enable live paper-trading phase before any real execution.
