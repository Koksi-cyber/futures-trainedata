"""
Utility functions to compute technical indicators for candlestick data.

This module provides simple implementations of common indicators such as
Exponential Moving Average (EMA) and Relative Strength Index (RSI).  It
avoids external dependencies other than NumPy and pandas so it can be used
in lightweight environments.  These indicators are used both for
feature engineering during model training and for real‑time signal
generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    """Return the exponential moving average of a pandas Series.

    Parameters
    ----------
    series : pd.Series
        The input series of values.
    span : int
        The span (period) for the EMA.

    Returns
    -------
    pd.Series
        The exponentially weighted moving average.
    """
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) of a price series.

    RSI oscillates between 0 and 100 and is based on average gains and
    losses over the specified window.  It is often used to identify
    overbought and oversold conditions.

    Parameters
    ----------
    series : pd.Series
        Series of close prices.
    period : int, optional
        Number of periods to use when calculating average gain and
        average loss.  Default is 14.

    Returns
    -------
    pd.Series
        RSI values aligned with the input series.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    # Avoid division by zero
    rs = avg_gain / avg_loss.replace({0: np.nan})
    rsi = 100 - (100 / (1 + rs))
    return rsi


def candle_body_strength(open_: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """Compute the ratio of the candle body size to its total range.

    The body strength is defined as (close ‑ open) / (high ‑ low).  A
    value above zero indicates a bullish candle and below zero a bearish
    candle.  Larger absolute values indicate strong momentum within the
    candle.  When the high and low are equal the denominator is set to
    1 to avoid division by zero.

    Parameters
    ----------
    open_ : pd.Series
        Series of open prices.
    close : pd.Series
        Series of close prices.
    high : pd.Series
        Series of high prices.
    low : pd.Series
        Series of low prices.

    Returns
    -------
    pd.Series
        Series of candle body strength values.
    """
    body = close - open_
    range_ = high - low
    # Replace zero ranges with one to avoid division by zero
    range_safe = range_.replace(0, 1)
    return body / range_safe