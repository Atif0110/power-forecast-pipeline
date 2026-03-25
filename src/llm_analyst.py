"""Groq-powered trading insights (production-grade)"""

from groq import Groq
from dotenv import load_dotenv
import os
import logging
import json
from datetime import datetime

load_dotenv()
logger = logging.getLogger(__name__)


def generate_trader_brief(results, config):
    """Generate trader-ready daily brief using model + trading outputs"""

    # ---------------- SAFE EXTRACTION ---------------- #
    mae_improvement = results.get("improvement_%", 0)
    fc_avg = results.get("fc_avg", 0)
    signal = results.get("signal", "NO TRADE")
    spread = results.get("spread", 0)
    uncertainty = results.get("uncertainty", 0)
    pnl = results.get("pnl", 0)

    curve_price = config.get("trading", {}).get("curve_price", "NA")

    api_key = os.getenv("GROQ_API_KEY")

    # ---------------- NO API KEY → SKIP ---------------- #
    if not api_key:
        logger.info("LLM skipped (no API key)")
        return (
            f"{signal} | Spread {spread:+.1f} vs uncertainty ±{uncertainty:.1f}. "
            f"PnL est: {pnl:.1f}. Monitor volatility."
        )

    try:
        client = Groq(api_key=api_key)

        # ---------------- PROMPT ---------------- #
        prompt = f"""
You are a power trader at Cobblestone Energy.

Inputs:
- Model improvement: {mae_improvement:.2f}%
- Forecast avg price: {fc_avg:.2f} EUR/MWh
- Curve price: {curve_price}
- Spread: {spread:.2f}
- Uncertainty: ±{uncertainty:.2f}
- Signal: {signal}
- Estimated PnL: {pnl:.2f}

Write EXACTLY 4 short sentences:
1. Key driver
2. Trade (direction + conviction)
3. Risk
4. Invalidation

Be concise, professional, no fluff.
"""

        # ---------------- MODEL (UPDATED) ---------------- #
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.1
        )

        brief = response.choices[0].message.content.strip()

        # ---------------- VALIDATION ---------------- #
        if not isinstance(brief, str) or len(brief) < 10:
            raise ValueError("Invalid LLM output")

        # ---------------- LOGGING ---------------- #
        os.makedirs("outputs/logs", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_data = {
            "timestamp": timestamp,
            "prompt": prompt,
            "response": brief,
            "model": "llama3-8b-8192",
            "tokens": getattr(response.usage, "total_tokens", "NA") if hasattr(response, "usage") else "NA",
        }

        with open(f"outputs/logs/llm_brief_{timestamp}.json", "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info("Trader brief generated")

        return brief

    except Exception as e:
        # ---------------- SMART FALLBACK ---------------- #
        fallback = (
            f"{signal} | Spread {spread:+.1f} vs uncertainty ±{uncertainty:.1f}. "
            f"Model improvement {mae_improvement:.1f}%. "
            f"Invalidate if spread reverses or volatility spikes."
        )

        logger.warning(f"LLM fallback used: {e}")
        return fallback


# ---------------- TEST ---------------- #
if __name__ == "__main__":
    results = {
        "improvement_%": 35,
        "fc_avg": 142,
        "signal": "LONG",
        "spread": 12,
        "uncertainty": 6,
        "pnl": 250,
    }

    config = {
        "trading": {"curve_price": 130}
    }

    brief = generate_trader_brief(results, config)
    print(brief)