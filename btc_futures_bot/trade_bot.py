"""
Real‑time trading bot for BTCUSDT perpetual futures.

This module defines the ``TradeBot`` class which encapsulates the
functionality required to fetch market data, compute features,
generate trading signals using a pre‑trained machine learning model,
and manage open positions.  The bot is designed to check for
opportunities every minute using multi‑timeframe data (1m, 5m, 1h and
4h) and only take a trade when the model predicts a high probability
of achieving at least a 1.2× reward before hitting the stop loss.

The current implementation focuses on long trades because the market
conditions during backtesting were predominantly bullish.  However,
the architecture allows for future extension to short trades by
training a separate model or inverting the feature set.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import joblib  # type: ignore
import numpy as np
import pandas as pd
import requests

from indicators import ema, rsi, candle_body_strength


BINANCE_BASE_URL = "https://fapi.binance.com"


def fetch_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Fetch recent candlesticks from Binance Futures API.

    Parameters
    ----------
    symbol : str
        Trading pair symbol, e.g. ``"BTCUSDT"``.
    interval : str
        Kline interval (``"1m"``, ``"5m"``, ``"1h"``, ``"4h"``, etc.).
    limit : int, optional
        Maximum number of candles to retrieve.  Binance allows up to
        1500 candles for most intervals.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``time``, ``open``, ``high``, ``low``,
        ``close`` and sorted chronologically.
    """
    url = f"{BINANCE_BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # Each kline: [open_time, open, high, low, close, volume, close_time, ...]
    records = []
    for k in data:
        records.append({
            "time": pd.to_datetime(k[0], unit="ms"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
        })
    df = pd.DataFrame(records)
    df.sort_values("time", inplace=True)
    return df


@dataclass
class TradeSignal:
    """Representation of a trading signal."""

    timestamp: _dt.datetime
    direction: str  # "LONG" or "SHORT"
    probability: float
    entry_price: float
    stop_loss: float
    take_profit: float


@dataclass
class TradeBot:
    """Bot that uses a pre‑trained model to generate BTC futures signals."""

    symbol: str = "BTCUSDT"
    # Probability threshold for taking a trade.  When using the
    # 1.5 R model, higher thresholds reduce the number of trades but
    # improve accuracy.  The default of 0.66 corresponds to roughly
    # 67 % success on the one‑year backtest.
    threshold: float = 0.66
    # Maximum 5‑minute RSI allowed for a long trade.  During training
    # samples were restricted to RSI values below 55; replicating the
    # same filter at runtime improves signal quality.
    rsi_threshold: float = 55.0
    model_path: Optional[str] = None
    model: Optional[object] = field(init=False, default=None)

    # Internal buffers for candle data
    candles_1m: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    candles_5m: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    candles_1h: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    candles_4h: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        # Load model on initialisation.  By default use the
        # 1.5 R model; fall back to the original model if not found.
        if self.model_path is None:
            # Derive default model path relative to this file
            default_15r = os.path.join(os.path.dirname(__file__), "models", "long_model_15R.pkl")
            default_12r = os.path.join(os.path.dirname(__file__), "models", "long_model.pkl")
            # Prefer 1.5 R model if it exists
            self.model_path = default_15r if os.path.isfile(default_15r) else default_12r
        self.load_model(self.model_path)

    def load_model(self, path: str) -> None:
        """Load a pickled sklearn model from disk."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------
    def refresh_candles(self) -> None:
        """Refresh cached candle data from Binance.

        This method fetches the most recent candles for 1m, 5m, 1h and
        4h intervals.  The default fetch limit of 500 provides
        sufficient history for computing EMAs and RSI.
        """
        self.candles_1m = fetch_klines(self.symbol, "1m", limit=500)
        self.candles_5m = fetch_klines(self.symbol, "5m", limit=500)
        self.candles_1h = fetch_klines(self.symbol, "1h", limit=500)
        self.candles_4h = fetch_klines(self.symbol, "4h", limit=500)
        # Compute indicators on the fetched data
        self.candles_5m["ema20"] = ema(self.candles_5m["close"], 20)
        self.candles_5m["ema50"] = ema(self.candles_5m["close"], 50)
        self.candles_5m["rsi"] = rsi(self.candles_5m["close"], 14)
        self.candles_1h["ema50"] = ema(self.candles_1h["close"], 50)
        self.candles_4h["ema50"] = ema(self.candles_4h["close"], 50)

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------
    def compute_current_features(self) -> Optional[List[float]]:
        """Compute the feature vector for the latest minute.

        Returns ``None`` if there is insufficient data to compute
        indicators or align the higher timeframes.
        """
        if self.candles_1m.empty or self.candles_5m.empty or self.candles_1h.empty or self.candles_4h.empty:
            return None
        # Use the most recent completed 1m candle as the reference
        latest_1m = self.candles_1m.iloc[-1]
        t = latest_1m["time"]
        # Align 5m, 1h and 4h frames using asof logic
        df5 = self.candles_5m[self.candles_5m["time"] <= t]
        df1h = self.candles_1h[self.candles_1h["time"] <= t]
        df4h = self.candles_4h[self.candles_4h["time"] <= t]
        if df5.empty or df1h.empty or df4h.empty:
            return None
        row5 = df5.iloc[-1]
        row1h = df1h.iloc[-1]
        row4h = df4h.iloc[-1]
        # Determine trend (must be up for now)
        if not ((row4h["close"] > row4h["ema50"]) and (row1h["close"] > row1h["ema50"])):
            return None
        # Ensure momentum condition
        # Ensure indicator values are present and not NaN
        if row5["ema20"] is None or row5["ema50"] is None or row5["rsi"] is None:
            return None
        # pandas may store missing values as NaN rather than None
        if pd.isna(row5["ema20"]) or pd.isna(row5["ema50"]) or pd.isna(row5["rsi"]):
            return None
        if row5["ema20"] <= row5["ema50"]:
            return None
        # Enforce RSI filter consistent with training
        if row5["rsi"] >= self.rsi_threshold:
            return None
        # Build feature vector consistent with the training script
        body_strength = candle_body_strength(pd.Series([latest_1m["open"]]), pd.Series([latest_1m["close"]]), pd.Series([latest_1m["high"]]), pd.Series([latest_1m["low"]])).iloc[0]
        features = [
            row5["rsi"],
            row5["ema20"] - row5["ema50"],
            (row1h["close"] - row1h["ema50"]) / row1h["ema50"] if row1h["ema50"] else 0.0,
            (row4h["close"] - row4h["ema50"]) / row4h["ema50"] if row4h["ema50"] else 0.0,
            (latest_1m["close"] - row5["ema50"]) / row5["ema50"] if row5["ema50"] else 0.0,
            body_strength,
        ]
        return features

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------
    def generate_signal(self) -> Optional[TradeSignal]:
        """Generate a trading signal if the model probability exceeds the threshold.

        Returns ``None`` if no trade should be taken at the current minute.
        """
        features = self.compute_current_features()
        if features is None or self.model is None:
            return None
        prob = float(self.model.predict_proba([features])[0][1])
        if prob < self.threshold:
            return None
        # Prepare signal details
        latest = self.candles_1m.iloc[-1]
        prev = self.candles_1m.iloc[-2] if len(self.candles_1m) >= 2 else latest
        entry = latest["close"]
        stop = prev["low"]
        risk = entry - stop
        if risk <= 0:
            return None
        # Compute the take‑profit using the 1.5 R multiplier.  This reflects
        # the updated requirement to seek a 50 % gain relative to the risk.
        take = entry + 1.5 * risk
        return TradeSignal(
            timestamp=latest["time"].to_pydatetime(),
            direction="LONG",
            probability=prob,
            entry_price=entry,
            stop_loss=stop,
            take_profit=take,
        )