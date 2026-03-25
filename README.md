# Power Price Forecasting & Trading Pipeline

An end-to-end pipeline for forecasting day-ahead electricity prices and translating forecasts into actionable trading signals.

---

## Overview

This project builds a complete workflow for power market analysis, including:

- Data ingestion and preprocessing
- Data quality assurance (QA)
- Machine learning-based price forecasting
- Translation of forecasts into trading signals
- PnL simulation
- Optional AI-assisted analysis for trader insights

The pipeline is designed to be robust, reproducible, and usable with or without external API access.

---

## Key Features

### 1. Data Ingestion
- Fetches day-ahead electricity price data
- Incorporates fundamental drivers such as:
  - Load (demand proxy)
  - Renewable generation (supply proxy)
- Includes a synthetic fallback dataset for reproducibility

---

### 2. Data Quality Assurance (QA)
- Time integrity checks (missing/duplicate timestamps)
- Statistical checks (distribution, outliers, anomalies)
- Economic sanity checks:
  - Price vs demand (positive relationship)
  - Price vs renewables (negative relationship)
- Generates a structured QA report

---

### 3. Forecasting
- Baseline: previous-day (lag-24) price
- Model: Gradient Boosting (XGBoost)
- Feature engineering includes:
  - Time-based features (hour, weekday)
  - Lag features (1h, 24h, 168h)
  - Rolling statistics and momentum

---

### 4. Trading Signal Generation
- Forecasts are converted into trading signals using:
  - Spread between forecasted price and market reference price
  - Uncertainty (confidence intervals)
- Signal logic:
  - LONG: forecast > reference + uncertainty
  - SHORT: forecast < reference - uncertainty
  - NO TRADE: within uncertainty band

---

### 5. PnL Simulation
- Simulates trading performance based on signals
- Tracks cumulative PnL over the forecast horizon

---

### 6. AI-Assisted Insights (Optional)
- Generates concise trading summaries
- Flags anomalies and potential risks
- Automatically skipped if API key is not provided

---

## Project Structure


.
├── src/
│ ├── data_ingestion.py
│ ├── qa_pipeline.py
│ ├── forecasting.py
│ ├── trading_view.py
│ └── llm_analyst.py
├── tests/
├── outputs/
├── main.py
├── config.yaml
├── requirements.txt
├── README.md
└── .env.example


---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
2. Run the pipeline
python main.py
Outputs

After execution, the following outputs are generated:

outputs/charts/trading_view.html
→ Interactive visualization of forecasts, signals, and PnL
outputs/submission.csv
→ Forecasted values with confidence intervals
outputs/logs/qa_report.json
→ Data quality report
Environment Variables (Optional)

Create a .env file:

ENTSOE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here

If not provided:

Synthetic data will be used
AI components will be skipped
