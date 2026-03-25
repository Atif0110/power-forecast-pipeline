"""
Cobblestone Power Forecasting Pipeline

End-to-end pipeline for:
- Power price forecasting (XGBoost)
- Data QA and validation
- Trading signal generation
- LLM-based trader insights

Designed for reproducibility, modularity, and trading relevance.
"""

# Version (useful for debugging / logs)
__version__ = "1.0.0"

# Public API (clean imports)
from .data_ingestion import fetch_de_power_data
from .qa_pipeline import run_qa
from .forecasting import train_forecast_model
from .trading_view import create_trading_view
from .llm_analyst import generate_trader_brief

# Define what gets imported with `from src import *`
__all__ = [
    "fetch_de_power_data",
    "run_qa",
    "train_forecast_model",
    "create_trading_view",
    "generate_trader_brief",
]