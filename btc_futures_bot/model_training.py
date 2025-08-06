"""
Script to train a machine learning model for BTCUSDT futures signal generation.

This script reads 90‑day historical candlestick data from multiple
timeframes (1m, 5m, 1h and 4h) provided by the `bactesting candles`
dataset.  It constructs a feature matrix and binary labels based on
whether a long trade would have achieved a 1.2× reward before
hitting the stop loss (defined as the low of the previous 1‑minute
candle).  A random forest classifier is trained to predict the
probability of trade success.  The resulting model is saved to
``models/long_model.pkl`` for use in the live bot.

To run the training:

    python model_training.py

Make sure that the ``bactesting candles`` directory resides in the
same parent directory as this script.  The trained model will be
stored alongside the script in a ``models`` subdirectory.
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


def load_dataframe(json_path: Path) -> pd.DataFrame:
    """Load a list of candlesticks from a JSON file into a DataFrame.

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


def build_feature_label_matrix(
    df1m: pd.DataFrame,
    df5m: pd.DataFrame,
    df1h: pd.DataFrame,
    df4h: pd.DataFrame,
    horizon: int = 20,
    rsi_threshold: float = 55.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a feature matrix and label vector for long trades.

    For each 1‑minute candle, this function determines whether the
    market is in an uptrend (based on 1h and 4h EMA50) and whether the
    short‑term momentum (EMA20 > EMA50 on the 5‑minute timeframe)
    warrants considering a long trade.  If these conditions are
    satisfied and the 5‑minute RSI is below ``rsi_threshold``, a feature
    vector is constructed.  The label is 1 if, within ``horizon``
    subsequent minutes, the price reaches a take‑profit level 1.2 times
    the risk (entry minus previous low) before hitting the stop loss
    (previous low).  Otherwise the label is 0.

    Parameters
    ----------
    df1m : pd.DataFrame
        DataFrame of 1‑minute candles.
    df5m : pd.DataFrame
        DataFrame of 5‑minute candles with EMA20, EMA50 and RSI already
        computed.
    df1h : pd.DataFrame
        DataFrame of 1‑hour candles with EMA50 computed.
    df4h : pd.DataFrame
        DataFrame of 4‑hour candles with EMA50 computed.
    horizon : int, optional
        Number of minutes to look ahead for the outcome.  Default is 20.
    rsi_threshold : float, optional
        Maximum 5‑minute RSI value to consider for a potential long
        trade.  Default is 55.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the feature matrix ``X`` of shape
        (n_samples, n_features) and the label vector ``y`` of length
        ``n_samples``.
    """
    # Align lower and higher timeframes to 1‑minute timestamps using asof joins
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

    merged = pd.merge_asof(df1m, feat5, on="time", direction="backward", tolerance=pd.Timedelta("5m"))
    merged = pd.merge_asof(merged, feat1h, on="time", direction="backward", tolerance=pd.Timedelta("1h"))
    merged = pd.merge_asof(merged, feat4h, on="time", direction="backward", tolerance=pd.Timedelta("4h"))

    # Identify uptrend: both 1h and 4h close above their EMA50
    merged["uptrend"] = (merged["close4h"] > merged["ema50_4h"]) & (merged["close1h"] > merged["ema50_1h"])

    # Candle body strength
    merged["body_strength"] = candle_body_strength(merged["open"], merged["close"], merged["high"], merged["low"])

    X: List[List[float]] = []
    y: List[int] = []

    # Iterate through each minute and build features/labels
    total_rows = len(merged)
    for idx, row in merged.iterrows():
        # Filter for potential long trade
        if not row["uptrend"]:
            continue
        if row["ema20_5m"] is None or row["ema50_5m"] is None:
            continue
        if row["ema20_5m"] <= row["ema50_5m"]:
            continue
        if row["rsi5m"] is None or row["rsi5m"] >= rsi_threshold:
            continue
        # Determine the index in the merged DataFrame
        i = merged.index.get_loc(idx)
        if i == 0:
            continue
        # Compute risk and targets based on previous bar's low
        entry_price = row["close"]
        prev_low = merged.iloc[i - 1]["low"]
        risk = entry_price - prev_low
        if risk <= 0:
            continue
        take_profit = entry_price + 1.2 * risk
        # Determine label by scanning forward up to horizon minutes
        label = 0
        for j in range(i + 1, min(i + 1 + horizon, total_rows)):
            high = merged.iloc[j]["high"]
            low = merged.iloc[j]["low"]
            if high >= take_profit:
                label = 1
                break
            if low <= prev_low:
                label = 0
                break
        # Build feature vector
        # Differences and ratios help the model generalise across price scales
        f_vec = [
            row["rsi5m"],
            row["ema20_5m"] - row["ema50_5m"],
            (row["close1h"] - row["ema50_1h"]) / row["ema50_1h"] if row["ema50_1h"] else 0.0,
            (row["close4h"] - row["ema50_4h"]) / row["ema50_4h"] if row["ema50_4h"] else 0.0,
            (row["close"] - row["ema50_5m"]) / row["ema50_5m"] if row["ema50_5m"] else 0.0,
            row["body_strength"],
        ]
        X.append(f_vec)
        y.append(label)

    return np.array(X), np.array(y)


def main() -> None:
    """Main entry point for training the long‑trade model."""
    # Determine the path to the bactesting candles directory relative to this file
    script_dir = Path(__file__).resolve().parent
    candles_dir = script_dir.parent / "future_repo" / "bactesting_candles" / "bactesting candles"
    # Some environments may place the dataset in different directories;
    # fallback to a sibling directory if needed
    if not candles_dir.exists():
        candles_dir = script_dir.parent / "bactesting candles"

    # Load timeframes
    df1m = load_dataframe(candles_dir / "BTCUSDT_1m_90d.json")
    df5m = load_dataframe(candles_dir / "BTCUSDT_5m_90d.json")
    df1h = load_dataframe(candles_dir / "BTCUSDT_1h_90d.json")
    df4h = load_dataframe(candles_dir / "BTCUSDT_4h_90d.json")

    # Compute indicators on higher frames
    df5m["ema20"] = ema(df5m["close"], 20)
    df5m["ema50"] = ema(df5m["close"], 50)
    df5m["rsi"] = rsi(df5m["close"], 14)
    df1h["ema50"] = ema(df1h["close"], 50)
    df4h["ema50"] = ema(df4h["close"], 50)

    # Build dataset
    X, y = build_feature_label_matrix(df1m, df5m, df1h, df4h)
    print(f"Feature matrix shape: {X.shape}, positive rate: {y.mean():.3f}")

    # Split into train and test to estimate performance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train a random forest classifier; tune hyperparameters as needed
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate on test set across different thresholds for selecting trades
    prob_test = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.5, 0.9, 21)
    print("Threshold | Trades | Success rate")
    for t in thresholds:
        preds = prob_test >= t
        if preds.sum() == 0:
            continue
        acc = (y_test[preds] == 1).sum() / preds.sum()
        print(f"{t:.2f}\t{preds.sum():5d}\t{acc:.3f}")

    # Fit model on the entire dataset before saving
    model.fit(X, y)
    # Ensure the models directory exists
    models_dir = script_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "long_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()