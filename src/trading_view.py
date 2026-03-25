"""Prompt curve translation with trading logic + PnL simulation"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


def create_trading_view(results, config):
    """Convert forecast to trading decision with risk logic"""

    # ---------------- SAFETY CHECK ---------------- #
    required_keys = ["preds", "actual", "ci_lower", "ci_upper"]
    for key in required_keys:
        if key not in results:
            raise ValueError(f"Missing key in results: {key}")

    preds = np.array(results["preds"])
    actual = np.array(results["actual"])
    ci_lower = np.array(results["ci_lower"])
    ci_upper = np.array(results["ci_upper"])

    if not (len(preds) == len(actual) == len(ci_lower) == len(ci_upper)):
        raise ValueError("Prediction arrays must have same length")

    # ---------------- CONFIG SAFE ---------------- #
    curve_price = config.get("trading", {}).get("curve_price", 0)

    # ---------------- CORE METRICS ---------------- #
    fc_avg = preds.mean()
    spread = fc_avg - curve_price
    uncertainty = (ci_upper - ci_lower).mean() / 2

    # ---------------- SIGNAL LOGIC ---------------- #
    if abs(spread) < uncertainty:
        signal = "NO TRADE"
        direction = 0
        color = "gray"

    elif spread > 0:
        signal = "LONG PROMPT"
        direction = 1
        color = "green"

    else:
        signal = "SHORT PROMPT"
        direction = -1
        color = "red"

    # ---------------- PnL SIMULATION ---------------- #
    pnl_series = direction * (actual - curve_price)
    cumulative_pnl = pd.Series(pnl_series).cumsum()

    # ---------------- INVALIDATION ---------------- #
    invalidation = (
        "Invalidate if spread flips OR volatility > expected"
    )

    # ---------------- TIME INDEX ---------------- #
    if "index" in results:
        time_index = results["index"]
    else:
        time_index = pd.date_range(
            end=pd.Timestamp.today(),
            periods=len(preds),
            freq="h"
        )

    # ---------------- PLOT ---------------- #
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Forecast vs Actual (with CI)",
            "Trading Signal",
            "PnL Simulation"
        ),
        vertical_spacing=0.08
    )

    # --- Forecast ---
    fig.add_trace(
        go.Scatter(x=time_index, y=actual, name="Actual"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=time_index, y=preds, name="Forecast"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=time_index, y=ci_upper, name="CI Upper", line=dict(dash="dash")),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=time_index,
            y=ci_lower,
            fill="tonexty",
            name="CI Lower"
        ),
        row=1, col=1
    )

    # --- Signal Box ---
    signal_text = (
        f"Forecast Avg: {fc_avg:.1f}€ | Curve: {curve_price}€ | Spread: {spread:+.1f}€<br>"
        f"Uncertainty: ±{uncertainty:.1f}€<br>"
        f"<b>{signal}</b><br>"
        f"Invalidation: {invalidation}"
    )

    fig.add_annotation(
        x=0.01,
        y=0.9,
        text=signal_text,
        showarrow=False,
        row=2,
        col=1
    )

    # --- PnL ---
    fig.add_trace(
        go.Scatter(x=time_index, y=cumulative_pnl, name="Cumulative PnL"),
        row=3, col=1
    )

    fig.update_layout(
        height=900,
        title="DE Power Fair Value to Trading View"
    )

    # ---------------- SAVE ---------------- #
    charts_path = config.get("paths", {}).get("charts", "outputs/charts/")
    predictions_path = config.get("paths", {}).get("predictions", "outputs/submission.csv")

    os.makedirs(charts_path, exist_ok=True)
    fig.write_html(os.path.join(charts_path, "trading_view.html"))

    # ---------------- SUBMISSION ---------------- #
    submission = pd.DataFrame({
        "id": range(len(preds)),
        "y_pred": np.round(preds, 2),
        "ci_lower": np.round(ci_lower, 2),
        "ci_upper": np.round(ci_upper, 2),
    })

    submission.to_csv(predictions_path, index=False)

    logger.info(
        f" Signal: {signal} | Spread: {spread:+.2f}€ | "
        f"PnL: {cumulative_pnl.iloc[-1]:+.2f}"
    )

    return {
        "signal": signal,
        "spread": round(spread, 2),
        "uncertainty": round(uncertainty, 2),
        "pnl": round(cumulative_pnl.iloc[-1], 2),
        "invalidation": invalidation,
    }


# ---------------- TEST ---------------- #
if __name__ == "__main__":
    results = {
        "preds": np.random.normal(85, 10, 168),
        "actual": np.random.normal(85, 10, 168),
        "ci_lower": np.random.normal(75, 5, 168),
        "ci_upper": np.random.normal(95, 5, 168),
    }

    config = {
        "trading": {"curve_price": 90},
        "paths": {
            "charts": "outputs/charts/",
            "predictions": "outputs/submission.csv",
        },
    }

    output = create_trading_view(results, config)
    print(output)