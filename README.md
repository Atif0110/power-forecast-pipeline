# European Power Fair Value Pipeline (Cobblestone Case Study)

## Overview
This project builds an end-to-end pipeline to forecast Day-Ahead (DA) power prices for the DE-LU market and translate forecasts into actionable prompt curve trading signals.

The pipeline combines:
- Time-series forecasting (XGBoost)
- Fundamental drivers (load, renewables)
- Data QA and anomaly detection
- Trading signal generation with risk controls
- LLM-based trader brief automation

---

## Pipeline Architecture


Data → QA → Forecast → Trading Signal → LLM Brief


Modules:
- `data_ingestion.py` → Fetch ENTSO-E or generate realistic fallback data
- `qa_pipeline.py` → Time integrity, statistical checks, anomaly detection
- `forecasting.py` → Baseline vs XGBoost forecasting with confidence intervals
- `trading_view.py` → Converts forecast into trading signals + PnL simulation
- `llm_analyst.py` → Generates trader-style daily brief using LLM

---

## Data

- Market: DE-LU (Germany/Luxembourg)
- Frequency: Hourly
- Target: Day-Ahead prices
- Drivers:
  - Load forecast (demand proxy)
  - Wind + solar generation (renewable supply proxy)

Fallback synthetic data is used if ENTSO-E API key is not provided.

---

## Forecasting Approach

### Baseline
- Lag-24 (yesterday same hour)

### Model
- XGBoost Regressor

### Features
- Time: hour, day-of-week, month
- Lags: 1h, 24h, 48h, 168h
- Rolling: mean, volatility
- Momentum: price change

### Evaluation
- MAE vs baseline
- Improvement %

---

## Trading Logic

Forecasts are translated into a prompt curve view:

- Compute spread = Forecast − Curve price
- Compare spread vs uncertainty (confidence interval)

### Signal Rules
- If |spread| < uncertainty → **No trade**
- If spread > uncertainty → **Long prompt**
- If spread < -uncertainty → **Short prompt**

### Risk Controls
- Confidence-based filtering
- Invalidation:
  - Spread reverses sign
  - Volatility exceeds expected range

### Backtest (Simplified)
- PnL simulated using actual vs curve price

---

## QA & Data Validation

- Missing timestamps detection
- Duplicate handling
- Outlier detection (z-score)
- Correlation sanity checks
- Economic validation:
  - Price ↑ with load
  - Price ↓ with renewables

Quality score assigned (0–100)

---

## LLM Integration

Groq (LLaMA3) used to generate a daily trader brief:
- Interprets model outputs
- Highlights key drivers
- Suggests trade + risks
- Logs prompt + response for reproducibility

Fallback logic included if API unavailable.

---

## How to Run

```bash
pip install -r requirements.txt
python main.py

Run tests:

python tests/test_pipeline.py
Outputs

Generated in outputs/:

data/ → raw dataset
charts/ → trading visualization
submission.csv → predictions
logs/ → QA report + LLM logs
Notes
Synthetic fallback ensures reproducibility without API dependency
Pipeline designed to be easily extendable to real ENTSO-E data
Focus is on trading relevance, not just prediction accuracy
Author

Mohd Atif
Email: data.atif001@gmail.com