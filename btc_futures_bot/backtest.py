"""
Backtesting script for the BTC futures trading bot.

This script loads the historical candlestick data provided in the
``bactesting candles`` dataset, computes the same features used for
training, applies the pre‑trained random forest model to generate
probabilities and selects trades when the probability exceeds a user
defined threshold.  Each selected trade is then evaluated using a
fixed risk (previous 1‑minute low) and a 1.2× reward target.  The
overall success rate and number of trades are reported.

Usage:

    python backtest.py --threshold 0.6

Adjust ``--threshold`` to control the trade selection confidence.  A
higher threshold yields fewer trades but a higher accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import joblib  # type: ignore
import numpy as np
import pandas as pd

from indicators import ema, rsi, candle_body_strength


def load_dataframe(json_path: Path) -> pd.DataFrame:
    with json_path.open() as f:
        data = json.load(f)
    df = pd.DataFrame({
        "time": pd.to_datetime([d["open_time"] for d in data]),
        "open": [float(d["open"]) for d in data],
        "high": [float(d["high"]) for d in data],
        "low": [float(d["low"]) for d in data],
        "close": [float(d["close"]) for d in data],
    })
    df.sort_values("time", inplace=True)
    return df


def build_merged_dataframe(df1m: pd.DataFrame, df5m: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame) -> pd.DataFrame:
    """Align 5m, 1h and 4h indicators to the 1m timeframe."""
    feat5 = df5m[["time", "ema20", "ema50", "rsi"]].rename(columns={
        "ema20": "ema20_5m",
        "ema50": "ema50_5m",
        "rsi": "rsi5m",
    })
    feat1 = df1h[["time", "ema50", "close"]].rename(columns={
        "ema50": "ema50_1h",
        "close": "close1h",
    })
    feat4 = df4h[["time", "ema50", "close"]].rename(columns={
        "ema50": "ema50_4h",
        "close": "close4h",
    })
    merged = pd.merge_asof(df1m, feat5, on="time", direction="backward", tolerance=pd.Timedelta("5m"))
    merged = pd.merge_asof(merged, feat1, on="time", direction="backward", tolerance=pd.Timedelta("1h"))
    merged = pd.merge_asof(merged, feat4, on="time", direction="backward", tolerance=pd.Timedelta("4h"))
    # Forward/backward fill missing higher timeframe indicators to avoid NaNs
    cols_to_fill = [
        "ema20_5m",
        "ema50_5m",
        "rsi5m",
        "ema50_1h",
        "close1h",
        "ema50_4h",
        "close4h",
    ]
    merged[cols_to_fill] = merged[cols_to_fill].fillna(method="ffill").fillna(method="bfill")
    # Compute uptrend flag
    merged["uptrend"] = (merged["close4h"] > merged["ema50_4h"]) & (merged["close1h"] > merged["ema50_1h"])
    # Candle body strength
    merged["body_strength"] = candle_body_strength(merged["open"], merged["close"], merged["high"], merged["low"])
    return merged


def compute_features_row(row: pd.Series) -> List[float]:
    """Compute the feature vector for a merged row consistent with training."""
    return [
        row["rsi5m"],
        row["ema20_5m"] - row["ema50_5m"],
        (row["close1h"] - row["ema50_1h"]) / row["ema50_1h"] if row["ema50_1h"] else 0.0,
        (row["close4h"] - row["ema50_4h"]) / row["ema50_4h"] if row["ema50_4h"] else 0.0,
        (row["close"] - row["ema50_5m"]) / row["ema50_5m"] if row["ema50_5m"] else 0.0,
        row["body_strength"],
    ]


def backtest_strategy(
    merged: pd.DataFrame,
    model,
    threshold: float = 0.6,
    horizon: int = 20,
    rsi_threshold: float = 55.0,
) -> Tuple[int, int, float]:
    """Run a backtest on the merged DataFrame.

    The backtest iterates through each 1‑minute row and considers a long trade
    only when the market is in an uptrend (both 4h and 1h close above their
    respective EMA50), the 5‑minute EMA20 is above the EMA50, and the 5‑minute
    RSI is below ``rsi_threshold``.  For each qualifying row the pre‑trained
    model generates a probability score, and a trade is taken if the score is
    greater than or equal to ``threshold``.

    Each trade uses the previous minute's low as the stop loss.  A fixed
    take‑profit 1.2× the risk (entry minus stop) is employed.  The trade is
    scanned forward for ``horizon`` minutes; if the price reaches the take
    profit before hitting the stop, the trade is counted as a success.

    Parameters
    ----------
    merged : pd.DataFrame
        DataFrame containing the 1‑minute data merged with higher timeframe
        indicators and precomputed feature columns.
    model : object
        A scikit‑learn classifier supporting ``predict_proba`` used to
        estimate the success probability of each trade.
    threshold : float, default 0.6
        Minimum predicted probability required to take a trade.  Higher
        thresholds reduce the number of trades but typically improve
        accuracy.
    horizon : int, default 20
        Number of minutes ahead to look for the outcome of the trade.
    rsi_threshold : float, default 55.0
        Maximum 5‑minute RSI value to consider for a long trade.  During
        training the dataset was restricted to RSI values below this
        threshold; applying the same filter here ensures consistency.

    Returns
    -------
    Tuple[int, int, float]
        A tuple ``(trades, successes, accuracy)`` summarising the total
        number of trades taken, the number of successful trades and the
        resulting accuracy.
    """
    successes = 0
    failures = 0
    total_rows = len(merged)
    for idx, row in merged.iterrows():
        # Only consider rows that qualify for long trades
        if not row["uptrend"]:
            continue
        # Skip rows with missing indicator values
        if row["ema20_5m"] is None or row["ema50_5m"] is None or row["rsi5m"] is None:
            continue
        # pandas may store missing values as NaN rather than None
        if pd.isna(row["ema20_5m"]) or pd.isna(row["ema50_5m"]) or pd.isna(row["rsi5m"]):
            continue
        if row["ema20_5m"] <= row["ema50_5m"]:
            continue
        # Apply RSI filter consistent with training
        if row["rsi5m"] >= rsi_threshold:
            continue
        # Compute model probability
        features = compute_features_row(row)
        prob = float(model.predict_proba([features])[0][1])
        if prob < threshold:
            continue
        # Determine trade outcome
        i = merged.index.get_loc(idx)
        if i == 0:
            continue
        entry_price = row["close"]
        prev_low = merged.iloc[i - 1]["low"]
        risk = entry_price - prev_low
        if risk <= 0:
            continue
        take_profit = entry_price + 1.2 * risk
        result = 0
        for j in range(i + 1, min(i + 1 + horizon, total_rows)):
            high = merged.iloc[j]["high"]
            low = merged.iloc[j]["low"]
            if high >= take_profit:
                result = 1
                break
            if low <= prev_low:
                result = 0
                break
        if result == 1:
            successes += 1
        else:
            failures += 1
    trades = successes + failures
    accuracy = successes / trades if trades > 0 else 0.0
    return trades, successes, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the BTC futures trading bot.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Probability threshold for taking a trade (default: 0.6)")
    parser.add_argument("--rsi_threshold", type=float, default=55.0, help="Maximum 5m RSI allowed for a long trade (default: 55.0)")
    parser.add_argument("--model", type=str, default=None, help="Path to the trained model file")
    args = parser.parse_args()
    # Determine paths
    script_dir = Path(__file__).resolve().parent
    candles_dir = script_dir.parent / "future_repo" / "bactesting_candles" / "bactesting candles"
    if not candles_dir.exists():
        candles_dir = script_dir.parent / "bactesting candles"
    # Load data
    df1m = load_dataframe(candles_dir / "BTCUSDT_1m_90d.json")
    df5m = load_dataframe(candles_dir / "BTCUSDT_5m_90d.json")
    df1h = load_dataframe(candles_dir / "BTCUSDT_1h_90d.json")
    df4h = load_dataframe(candles_dir / "BTCUSDT_4h_90d.json")
    # Compute indicators
    df5m["ema20"] = ema(df5m["close"], 20)
    df5m["ema50"] = ema(df5m["close"], 50)
    df5m["rsi"] = rsi(df5m["close"], 14)
    df1h["ema50"] = ema(df1h["close"], 50)
    df4h["ema50"] = ema(df4h["close"], 50)
    # Merge
    merged = build_merged_dataframe(df1m, df5m, df1h, df4h)
    # Load model
    model_path: str
    if args.model is not None:
        model_path = args.model
    else:
        model_path = os.path.join(script_dir, "models", "long_model.pkl")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    # Run backtest
    trades, successes, accuracy = backtest_strategy(
        merged,
        model,
        threshold=args.threshold,
        horizon=20,
        rsi_threshold=args.rsi_threshold,
    )
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Total trades: {trades}")
    print(f"Successful trades: {successes}")
    print(f"Failed trades: {trades - successes}")
    print(f"Accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()