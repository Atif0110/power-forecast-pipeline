"""Comprehensive QA with time integrity + LLM anomaly detection"""

import pandas as pd
import numpy as np
import json
from groq import Groq
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logger = logging.getLogger(__name__)


def run_qa(df, config):
    """Full QA checks including time integrity + LLM analysis"""

    # ---------------- SAFETY CHECK ---------------- #
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty")

    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "dataset_shape": df.shape,
        "data_range": f"{df.index.min()} to {df.index.max()}",
    }

    # ================= TIME INTEGRITY ================= #
    expected = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing = expected.difference(df.index)

    report["time_checks"] = {
        "missing_hours": int(len(missing)),
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "is_monotonic": bool(df.index.is_monotonic_increasing),
    }

    # ================= NULLS ================= #
    report["nulls"] = df.isnull().sum().to_dict()

    # ================= PRICE STATS ================= #
    if "price" not in df.columns:
        raise ValueError("Missing 'price' column in data")

    price_stats = df["price"].describe()

    report["price_stats"] = {
        "mean": round(price_stats["mean"], 2),
        "std": round(price_stats["std"], 2),
        "min": round(price_stats["min"], 2),
        "max": round(price_stats["max"], 2),
        "negatives": int((df["price"] < 0).sum()),
        "extreme_spikes": int((df["price"] > 500).sum()),
    }

    # ================= OUTLIERS ================= #
    numeric_df = df.select_dtypes(include=[np.number])
    z = (numeric_df - numeric_df.mean()) / numeric_df.std()

    report["outliers"] = (abs(z) > 3).sum().to_dict()

    # ================= CORRELATION ================= #
    safe_cols = [c for c in ["price", "load_forecast", "wind_solar"] if c in df.columns]
    corrs = df[safe_cols].corr().get("price", pd.Series()).round(3)

    report["correlations"] = corrs.to_dict()

    # ================= ECONOMIC SANITY ================= #
    econ_checks = {
        "price_vs_load_positive": corrs.get("load_forecast", 0) > 0,
        "price_vs_renewables_negative": corrs.get("wind_solar", 0) < 0,
    }

    report["economic_checks"] = econ_checks

    # ================= QUALITY SCORE ================= #
    quality_score = 100

    if report["time_checks"]["missing_hours"] > 0:
        quality_score -= 20

    if report["time_checks"]["duplicate_timestamps"] > 0:
        quality_score -= 10

    if report["price_stats"]["extreme_spikes"] > 10:
        quality_score -= 10

    if not econ_checks["price_vs_load_positive"]:
        quality_score -= 20

    if not econ_checks["price_vs_renewables_negative"]:
        quality_score -= 20

    report["quality_score"] = max(0, quality_score)

    # ================= LLM ANALYSIS ================= #
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        report["llm_analysis"] = {
            "anomalies": [],
            "trading_note": "LLM skipped (no API key)",
            "quality": "unknown",
        }
    else:
        try:
            client = Groq(api_key=api_key)

            sample_cols = [c for c in ["price", "load_forecast", "wind_solar"] if c in df.columns]

            sample = (
                df[sample_cols]
                .tail(24)
                .round(2)
                .to_dict()
            )

            prompt = f"""
You are a power trader.

Analyze this last 24h data:
{sample}

Return STRICT JSON ONLY:
{{
    "anomalies": ["..."],
    "trading_note": "...",
    "quality": "good/medium/poor"
}}
"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            content = response.choices[0].message.content

            try:
                llm_result = json.loads(content)
            except Exception:
                llm_result = {
                    "anomalies": [],
                    "trading_note": content[:200],
                    "quality": "unknown",
                }

            report["llm_analysis"] = llm_result
            logger.info("LLM QA complete")

        except Exception as e:
            report["llm_analysis"] = {
                "anomalies": [],
                "trading_note": "LLM failed - manual review recommended",
                "quality": "unknown",
            }
            logger.warning(f"LLM QA skipped: {e}")

    # ================= SAVE ================= #
    os.makedirs("outputs/logs", exist_ok=True)

    with open(config["paths"]["qa_report"], "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"QA report saved | Quality Score: {report['quality_score']}")

    return report


# ================= TEST ================= #
if __name__ == "__main__":
    df = pd.read_csv("outputs/data/raw.csv", index_col=0, parse_dates=True)

    config = {"paths": {"qa_report": "outputs/logs/qa_report.json"}}

    qa = run_qa(df, config)

    print(json.dumps(qa, indent=2))