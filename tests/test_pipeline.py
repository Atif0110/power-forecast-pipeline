"""
End-to-end pipeline tests for Cobblestone case study
Run with: python tests/test_pipeline.py
"""

import os
import sys
import pandas as pd

# Allow imports from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ingestion import fetch_de_power_data
from src.qa_pipeline import run_qa
from src.forecasting import train_forecast_model
from src.trading_view import create_trading_view
from src.llm_analyst import generate_trader_brief


# ---------------- TEST CONFIG ---------------- #
CONFIG = {
    'data': {
        'market': 'DE_LU',
        'start': '2025-01-01',
        'end': '2025-03-01'
    },
    'model': {
        'test_split': 0.2,
        'n_estimators': 50
    },
    'trading': {
        'curve_price': 90
    },
    'paths': {
        'raw_data': 'outputs/data/test_raw.csv',
        'model_file': 'outputs/test_model.pkl',
        'charts': 'outputs/charts/',
        'predictions': 'outputs/test_submission.csv',
        'qa_report': 'outputs/logs/test_qa.json'
    }
}


# ---------------- GLOBAL DATA (shared safely) ---------------- #
def get_test_data():
    """Ensures data exists before any test"""
    if not os.path.exists(CONFIG['paths']['raw_data']):
        return fetch_de_power_data(CONFIG)
    return pd.read_csv(CONFIG['paths']['raw_data'], index_col=0, parse_dates=True)


# ---------------- TEST FUNCTIONS ---------------- #

def test_data_ingestion():
    print("\n[TEST] Data Ingestion")

    df = fetch_de_power_data(CONFIG)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'price' in df.columns

    print(" Data ingestion passed")


def test_qa():
    print("\n[TEST] QA Pipeline")

    df = get_test_data()
    report = run_qa(df, CONFIG)

    assert 'quality_score' in report
    assert report['dataset_shape'][0] > 0

    print(" QA passed")


def test_forecasting():
    print("\n[TEST] Forecasting")

    df = get_test_data()
    results = train_forecast_model(df, CONFIG)

    assert 'preds' in results
    assert len(results['preds']) > 0
    assert results['mae_xgb'] >= 0

    #  critical check
    assert not pd.isnull(results['preds']).any()

    print(" Forecasting passed")


def test_trading():
    print("\n[TEST] Trading View")

    df = get_test_data()
    results = train_forecast_model(df, CONFIG)

    trading_output = create_trading_view(results, CONFIG)

    assert 'signal' in trading_output
    assert 'spread' in trading_output

    print(" Trading passed")


def test_llm():
    print("\n[TEST] LLM Analyst")

    mock_results = {
        'improvement_%': 20,
        'fc_avg': 100,
        'signal': 'LONG',
        'spread': 10,
        'uncertainty': 5,
        'pnl': 50
    }

    brief = generate_trader_brief(mock_results, CONFIG)

    assert isinstance(brief, str)
    assert len(brief) > 0

    print(" LLM passed")


def test_full_pipeline():
    """🔥 Full integration test (VERY IMPORTANT)"""
    print("\n[TEST] Full Pipeline Integration")

    df = fetch_de_power_data(CONFIG)
    qa = run_qa(df, CONFIG)
    forecast = train_forecast_model(df, CONFIG)
    trading = create_trading_view(forecast, CONFIG)

    combined = {**forecast, **trading}
    brief = generate_trader_brief(combined, CONFIG)

    assert isinstance(brief, str)
    assert 'signal' in trading

    print(" Full pipeline passed")


# ---------------- RUN ALL ---------------- #

if __name__ == "__main__":
    print("\n=== RUNNING FULL PIPELINE TESTS ===")

    test_data_ingestion()
    test_qa()
    test_forecasting()
    test_trading()
    test_llm()
    test_full_pipeline()

    print("\n ALL TESTS PASSED SUCCESSFULLY")