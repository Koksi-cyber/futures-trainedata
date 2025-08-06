# features.py

import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from binance_api import get_klines

# === Constants ===
FEATURE_COLUMNS = [
    'ema_ratio', 'rsi_1m', 'rsi_5m', 'candle_body_pct_5m', 'price_above_1h_ema', 'slope_4h_ema'
]

# === Core Feature Builder ===
def extract_features(symbol="BTCUSDT"):
    try:
        # === 1m candles ===
        candles_1m = get_klines(symbol, "1m", limit=1000)
        df_1m = build_dataframe(candles_1m)

        # === 5m candles ===
        candles_5m = get_klines(symbol, "5m", limit=1000)
        df_5m = build_dataframe(candles_5m)

        # === 1h candles ===
        candles_1h = get_klines(symbol, "1h", limit=1000)
        df_1h = build_dataframe(candles_1h)

        # === 4h candles ===
        candles_4h = get_klines(symbol, "4h", limit=1000)
        df_4h = build_dataframe(candles_4h)

        # === Feature 1: EMA Ratio (1m) ===
        ema20 = EMAIndicator(df_1m['close'], window=20).ema_indicator()
        ema50 = EMAIndicator(df_1m['close'], window=50).ema_indicator()
        ema_ratio = ema20.iloc[-1] / ema50.iloc[-1] if ema50.iloc[-1] != 0 else 0

        # === Feature 2: RSI on 1m ===
        rsi_1m = RSIIndicator(df_1m['close'], window=14).rsi().iloc[-1]

        # === Feature 3: RSI on 5m ===
        rsi_5m = RSIIndicator(df_5m['close'], window=14).rsi().iloc[-1]

        # === Feature 4: Candle Body % (5m) ===
        last_5m = df_5m.iloc[-1]
        body = abs(last_5m['close'] - last_5m['open'])
        range_ = last_5m['high'] - last_5m['low']
        candle_body_pct_5m = body / range_ if range_ != 0 else 0

        # === Feature 5: Price above 1h EMA50 ===
        ema_1h_50 = EMAIndicator(df_1h['close'], window=50).ema_indicator()
        price_above_1h_ema = df_1h['close'].iloc[-1] > ema_1h_50.iloc[-1]

        # === Feature 6: Slope of 4h EMA20 ===
        ema_4h_20 = EMAIndicator(df_4h['close'], window=20).ema_indicator()
        slope_4h_ema = ema_4h_20.iloc[-1] - ema_4h_20.iloc[-5]  # difference over 5 candles

        # === Final Feature Vector ===
        features = pd.DataFrame([{
            'ema_ratio': ema_ratio,
            'rsi_1m': rsi_1m,
            'rsi_5m': rsi_5m,
            'candle_body_pct_5m': candle_body_pct_5m,
            'price_above_1h_ema': int(price_above_1h_ema),
            'slope_4h_ema': slope_4h_ema
        }], columns=FEATURE_COLUMNS)

        return features

    except Exception as e:
        print("[ERROR] Failed to extract features:", e)
        return None

# === Candle DataFrame Builder ===
def build_dataframe(klines):
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    return df
