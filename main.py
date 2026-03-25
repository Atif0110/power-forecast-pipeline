#!/usr/bin/env python3
"""
Cobblestone Energy - DE Power Fair Value Pipeline
Run: python main.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import yaml
import logging
from pathlib import Path

# Clean package imports
from src.data_ingestion import fetch_de_power_data
from src.qa_pipeline import run_qa
from src.forecasting import train_forecast_model
from src.trading_view import create_trading_view
from src.llm_analyst import generate_trader_brief


def load_config():
    """Load pipeline config safely"""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(" config.yaml missing - please create it")
        exit(1)


def setup_logging():
    """Setup logging once for entire pipeline"""
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("outputs/logs/pipeline.log"),
            logging.StreamHandler()
        ]
    )


def main():
    print(" Cobblestone DE-LU Power Pipeline Starting...\n")

    setup_logging()
    config = load_config()

    # ---------------- STEP 1 ---------------- #
    print("1/5  Loading power data...")
    df = fetch_de_power_data(config)

    # ---------------- STEP 2 ---------------- #
    print("2/5  Running QA checks...")
    qa_report = run_qa(df, config)

    # ---------------- STEP 3 ---------------- #
    print("3/5  Training forecasting model...")
    forecast_results = train_forecast_model(df, config)

    # ---------------- STEP 4 ---------------- #
    print("4/5  Generating trading signals...")
    trading_results = create_trading_view(forecast_results, config)

    # ---------------- STEP 5 ---------------- #
    print("5/5  Generating trader brief...")
    final_results = {**forecast_results, **trading_results}
    trader_brief = generate_trader_brief(final_results, config)

    # ---------------- OUTPUT ---------------- #
    print("\n" + "=" * 50)
    print(" TRADER BRIEF:")
    print(trader_brief)
    print("=" * 50)

    print("\n PIPELINE COMPLETE!")
    print(" outputs/charts/ to trading visualization")
    print(" outputs/submission.csv to predictions")
    print(" outputs/logs/qa_report.json to QA report")


if __name__ == "__main__":
    main()