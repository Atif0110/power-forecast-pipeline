"""XGBoost power price forecasting with robust validation + uncertainty"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import logging
import os

logger = logging.getLogger(__name__)


# ---------------- FEATURE ENGINEERING ---------------- #
def prepare_features(df):
    """Feature engineering aligned with power market dynamics"""

    df = df.copy()

    # ---------- TIME FEATURES ----------
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # ---------- LAGS ----------
    df["lag1"] = df["price"].shift(1)
    df["lag24"] = df["price"].shift(24)
    df["lag48"] = df["price"].shift(48)
    df["lag168"] = df["price"].shift(168)

    # ---------- ROLLING ----------
    df["price_ma24"] = df["price"].rolling(24).mean()
    df["price_std24"] = df["price"].rolling(24).std()

    # ---------- MOMENTUM ----------
    df["price_diff"] = df["price"] - df["lag24"]

    cols = [
        "load_forecast", "wind_solar",
        "hour", "dow", "month", "is_weekend",
        "lag1", "lag24", "lag48", "lag168",
        "price_ma24", "price_std24", "price_diff"
    ]

    df = df[cols + ["price"]].dropna()

    # 🔥 safety check
    if df.isnull().any().any():
        raise ValueError("NaNs present after feature engineering")

    return df


# ---------------- TRAINING ---------------- #
def train_forecast_model(df, config):
    """Train + validate + robust confidence intervals"""

    # ---------- SAFE CONFIG ----------
    model_cfg = config.get("model", {})
    test_split = model_cfg.get("test_split", 0.2)
    n_estimators = model_cfg.get("n_estimators", 200)

    Xy = prepare_features(df)

    X = Xy.drop("price", axis=1)
    y = Xy["price"]

    split = int(len(X) * (1 - test_split))

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ---------- BASELINE (aligned) ----------
    baseline = y.shift(24)
    baseline_test = baseline.iloc[split:]

    valid_idx = baseline_test.dropna().index

    mae_baseline = mean_absolute_error(
        y_test.loc[valid_idx],
        baseline_test.loc[valid_idx]
    )

    # ---------- MODEL ----------
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae_xgb = mean_absolute_error(y_test, y_pred)

    # ---------- UNCERTAINTY ----------
    train_pred = model.predict(X_train)
    train_residuals = y_train - train_pred

    ci_width = 1.96 * train_residuals.std()

    ci_lower = y_pred - ci_width
    ci_upper = y_pred + ci_width

    # ---------- SAVE MODEL ----------
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, config["paths"]["model_file"])

    # ---------- RESULTS ----------
    results = {
        "mae_baseline": round(mae_baseline, 2),
        "mae_xgb": round(mae_xgb, 2),
        "improvement_%": round(
            (mae_baseline - mae_xgb) / mae_baseline * 100, 2
        ),
        "ci_width": round(ci_width, 2),
        "preds": y_pred,
        "actual": y_test.values,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "index": y_test.index  #  important for downstream
    }

    logger.info(
        f" Model MAE: {results['mae_xgb']} | "
        f"Baseline: {results['mae_baseline']} | "
        f"Improvement: {results['improvement_%']}%"
    )

    return results


# ---------------- TEST ---------------- #
if __name__ == "__main__":
    df = pd.read_csv("outputs/data/raw.csv", index_col=0, parse_dates=True)

    config = {
        "model": {"test_split": 0.2, "n_estimators": 200},
        "paths": {"model_file": "outputs/model.pkl"}
    }

    results = train_forecast_model(df, config)

    print(results)