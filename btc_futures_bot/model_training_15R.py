"""
Script to train a machine‑learning model for BTCUSDT futures using a
1.5× reward‑to‑risk target.

This script mirrors the original ``model_training.py`` but changes
the take‑profit logic to 1.5 R instead of 1.2 R.  It loads 90‑day
candlestick data from multiple timeframes (1‑minute, 5‑minute,
1‑hour and 4‑hour), constructs a feature matrix describing the
market state when a potential long trade arises, and labels each
candidate entry as success or failure based on whether the price
reaches a 1.5 × reward before hitting the stop (previous low) within
20 minutes.

After training a random‑forest classifier, the script prints a
summary of accuracy at various probability thresholds on a held‑out
test set.  The trained model is saved to ``models/long_model_15R.pkl``.

Usage::

    python model_training_15R.py

Ensure the ``bactesting_candles`` directory is available in the
expected location before running.  Training may take several minutes
depending on hardware.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import joblib  # type: ignore
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from indicators import ema, rsi, candle_body_strength


def load_candles(json_path: Path) -> pd.DataFrame:
    """Load Binance kline JSON into a DataFrame.

    Parameters
    ----------
    json_path : Path
        Path to the JSON file containing an array of candle objects.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``time``, ``open``, ``high``, ``low`` and
        ``close`` sorted chronologically.
    """
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


def build_dataset(
    df1m: pd.DataFrame,
    df5m: pd.DataFrame,
    df1h: pd.DataFrame,
    df4h: pd.DataFrame,
    horizon: int = 20,
    rsi_threshold: float = 55.0,
    tp_multiplier: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a feature matrix and label vector for a 1.5 R long strategy.

    See ``model_training.py`` for a detailed description of the feature
    engineering.  The only differences are the take‑profit multiplier
    (``tp_multiplier``) and the horizon used to compute the outcome.

    Parameters
    ----------
    df1m : pd.DataFrame
        DataFrame of 1‑minute candles.
    df5m : pd.DataFrame
        DataFrame of 5‑minute candles with EMA20, EMA50 and RSI computed.
    df1h : pd.DataFrame
        DataFrame of 1‑hour candles with EMA50 computed.
    df4h : pd.DataFrame
        DataFrame of 4‑hour candles with EMA50 computed.
    horizon : int, optional
        Look‑ahead window in minutes for determining trade outcome.
        Default is 20 minutes.
    rsi_threshold : float, optional
        Maximum 5‑minute RSI value to consider for a potential long
        trade.  Default is 55.
    tp_multiplier : float, optional
        Reward‑to‑risk multiplier for the take‑profit level.  Default
        is 1.5.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the feature matrix ``X`` and the label
        vector ``y``.
    """
    # Align timeframes to 1‑minute using asof joins
    feat5 = df5m[["time", "ema20", "ema50", "rsi"]].rename(columns={
        "ema20": "ema20_5m",
        "ema50": "ema50_5m",
        "rsi": "rsi5m",
    })
    feat1h = df1h[["time", "ema50", "close"]].rename(columns={
        "ema50": "ema50_1h",
        "close": "close1h",
    })
    feat4h = df4h[["time", "ema50", "close"]].rename(columns={
        "ema50": "ema50_4h",
        "close": "close4h",
    })
    merged = pd.merge_asof(df1m, feat5, on="time", direction="backward",
                           tolerance=pd.Timedelta("5m"))
    merged = pd.merge_asof(merged, feat1h, on="time", direction="backward",
                           tolerance=pd.Timedelta("1h"))
    merged = pd.merge_asof(merged, feat4h, on="time", direction="backward",
                           tolerance=pd.Timedelta("4h"))

    # Uptrend filter
    merged["uptrend"] = (merged["close4h"] > merged["ema50_4h"]) & (
        merged["close1h"] > merged["ema50_1h"]
    )

    # Candle body strength
    merged["body_strength"] = candle_body_strength(
        merged["open"], merged["close"], merged["high"], merged["low"]
    )

    X: List[List[float]] = []
    y: List[int] = []
    total_rows = len(merged)
    # Iterate through each minute and build features/labels
    for idx, row in merged.iterrows():
        # Potential long trade filter
        if not row["uptrend"]:
            continue
        if pd.isna(row["ema20_5m"]) or pd.isna(row["ema50_5m"]):
            continue
        if row["ema20_5m"] <= row["ema50_5m"]:
            continue
        if pd.isna(row["rsi5m"]) or row["rsi5m"] >= rsi_threshold:
            continue
        # Determine DataFrame index position
        i = merged.index.get_loc(idx)
        # Skip first candle
        if i == 0:
            continue
        entry_price = row["close"]
        prev_low = merged.iloc[i - 1]["low"]
        risk = entry_price - prev_low
        if risk <= 0:
            continue
        take_profit = entry_price + tp_multiplier * risk
        # Evaluate outcome
        outcome = 0
        for j in range(i + 1, min(i + 1 + horizon, total_rows)):
            high = merged.iloc[j]["high"]
            low = merged.iloc[j]["low"]
            if high >= take_profit:
                outcome = 1
                break
            if low <= prev_low:
                outcome = 0
                break
        # Build feature vector
        f_vec = [
            row["rsi5m"],
            row["ema20_5m"] - row["ema50_5m"],
            (row["close1h"] - row["ema50_1h"]) / row["ema50_1h"] if row["ema50_1h"] else 0.0,
            (row["close4h"] - row["ema50_4h"]) / row["ema50_4h"] if row["ema50_4h"] else 0.0,
            (row["close"] - row["ema50_5m"]) / row["ema50_5m"] if row["ema50_5m"] else 0.0,
            row["body_strength"],
        ]
        X.append(f_vec)
        y.append(outcome)
    return np.array(X), np.array(y)


def main() -> None:
    """Train a random‑forest model for the 1.5 R long strategy."""
    # Locate dataset directory relative to this file
    script_dir = Path(__file__).resolve().parent
    candles_dir = script_dir.parent / "future_repo" / "bactesting_candles" / "bactesting candles"
    if not candles_dir.exists():
        # Fallback to ``bactesting candles`` in parent directory
        candles_dir = script_dir.parent / "bactesting candles"
    # Load candles
    print("Loading candles ...")
    df1m = load_candles(candles_dir / "BTCUSDT_1m_90d.json")
    df5m = load_candles(candles_dir / "BTCUSDT_5m_90d.json")
    df1h = load_candles(candles_dir / "BTCUSDT_1h_90d.json")
    df4h = load_candles(candles_dir / "BTCUSDT_4h_90d.json")
    # Compute indicators on higher timeframes
    print("Computing indicators ...")
    df5m["ema20"] = ema(df5m["close"], 20)
    df5m["ema50"] = ema(df5m["close"], 50)
    df5m["rsi"] = rsi(df5m["close"], 14)
    df1h["ema50"] = ema(df1h["close"], 50)
    df4h["ema50"] = ema(df4h["close"], 50)
    # Build dataset
    print("Constructing feature matrix and labels ...")
    X, y = build_dataset(df1m, df5m, df1h, df4h, tp_multiplier=1.5)
    print(f"Total samples: {len(y)} (positive: {np.sum(y)}, negative: {len(y) - np.sum(y)})")
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Train model
    print("Training random forest ...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    # Evaluate on test set with varying thresholds
    probs = clf.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.5, 0.8, 0.02)
    print("Threshold\tTrades\tAccuracy")
    for th in thresholds:
        mask = probs >= th
        n_trades = mask.sum()
        if n_trades > 0:
            acc = (y_test[mask] == 1).mean()
        else:
            acc = None
        print(f"{th:.2f}\t{n_trades}\t{acc}")
    # Persist model
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(clf, models_dir / "long_model_15R.pkl")
    print("Model saved to", models_dir / "long_model_15R.pkl")


if __name__ == "__main__":
    main()