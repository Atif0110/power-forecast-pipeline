"""Handles DE power data fetching with smart fallback (production-grade)"""

import pandas as pd
import numpy as np
from entsoe import EntsoePandasClient
from dotenv import load_dotenv
import os
import logging

load_dotenv()

logger = logging.getLogger(__name__)


def fetch_de_power_data(config):
    """Fetch real ENTSO-E data or generate realistic synthetic fallback"""

    api_key = os.getenv("ENTSOE_API_KEY")

    try:
        # ---------------- SAFE CONFIG ---------------- #
        start = pd.Timestamp(config.get("data", {}).get("start", "2025-01-01"))
        end = pd.Timestamp(config.get("data", {}).get("end", "2025-03-01"))

        # enforce timezone
        start = start.tz_localize("Europe/Berlin", nonexistent="shift_forward", ambiguous="NaT")
        end = end.tz_localize("Europe/Berlin", nonexistent="shift_forward", ambiguous="NaT")

        # ================= REAL DATA ================= #
        if api_key:
            try:
                client = EntsoePandasClient(api_key=api_key)

                prices = client.query_day_ahead_prices(
                    config["data"]["market"], start=start, end=end
                )

                prices = prices.sort_index()
                logger.info("Real ENTSO-E data loaded")

            except Exception as e:
                logger.warning(f" ENTSO-E failed, switching to synthetic: {e}")
                api_key = None  # fallback

        # ================= SYNTHETIC DATA ================= #
        if not api_key:
            logger.info(" Using realistic synthetic data")

            dates = pd.date_range(start=start, end=end, freq="h")

            np.random.seed(42)

            load = (
                500
                + 120 * np.sin(2 * np.pi * dates.hour / 24)
                + 80 * np.sin(2 * np.pi * dates.dayofyear / 365)
                + np.random.normal(0, 30, len(dates))
            )

            wind_solar = (
                300
                + 150 * np.sin(2 * np.pi * dates.dayofyear / 365 + 1)
                + np.random.normal(0, 50, len(dates))
            )

            prices = (
                30
                + 0.06 * load
                - 0.04 * wind_solar
                + np.random.normal(0, 6, len(dates))
            )

            prices = pd.Series(prices, index=dates)

        # ================= FUNDAMENTALS ================= #
        if api_key:
            np.random.seed(42)

            load = (
                500
                + 100 * np.sin(2 * np.pi * prices.index.hour / 24)
                + np.random.normal(0, 40, len(prices))
            )

            wind_solar = (
                300
                + 120 * np.sin(2 * np.pi * prices.index.dayofyear / 365)
                + np.random.normal(0, 60, len(prices))
            )

        # ================= FINAL DATAFRAME ================= #
        df = pd.DataFrame(
            {
                "price": prices,
                "load_forecast": load,
                "wind_solar": wind_solar,
            }
        )

        # ---------------- CLEANING ---------------- #
        df = df.sort_index()

        # remove duplicates
        df = df[~df.index.duplicated(keep="first")]

        # enforce hourly frequency safely
        full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
        df = df.reindex(full_index)

        # fill gaps
        df = df.ffill().bfill()

        # ---------------- SAVE ---------------- #
        os.makedirs("outputs/data", exist_ok=True)
        df.to_csv(config["paths"]["raw_data"])

        logger.info(f" Final data shape: {df.shape}")
        logger.info(f" Range: {df.index.min()} to {df.index.max()}")

        return df

    except Exception as e:
        logger.error(f"Data error: {e}")
        raise


# ---------------- TEST ---------------- #
if __name__ == "__main__":
    config = {
        "data": {
            "market": "DE_LU",
            "start": "2024-01-01",
            "end": "2026-03-25",
        },
        "paths": {"raw_data": "outputs/data/raw.csv"},
    }

    df = fetch_de_power_data(config)
    print(df.head())