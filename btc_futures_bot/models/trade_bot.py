# trade_bot.py

import time
import joblib
import os
from features import extract_features
from binance_api import place_market_order, get_account_balance, place_sl_tp_orders
import pandas as pd
from datetime import datetime

MODEL_PATH = os.path.join("models", "long_model_15R.pkl")
PROB_THRESHOLD = 0.66  # Minimum probability to take a trade
TRADE_SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSDT")
LEVERAGE = int(os.getenv("LEVERAGE", 10))
RISK_PER_TRADE = float(os.getenv("RISK_USDT", 10))

# === Load trained model ===
model = joblib.load(MODEL_PATH)

# === Log file ===
LOG_PATH = os.path.join("data", "trade_log.csv")
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, 'w') as f:
        f.write("timestamp,symbol,probability,decision,entry_price,sl,tp\n")

# === Trade Loop ===
print("📈 Bot started. Scanning every 60 seconds...")
while True:
    try:
        # === 1. Extract features ===
        features = extract_features(TRADE_SYMBOL)
        if features is None:
            time.sleep(60)
            continue

        # === 2. Predict probability ===
        prob = model.predict_proba(features)[0][1]  # Class 1 = positive outcome
        print(f"🧠 Prediction prob: {prob:.2%}")

        # === 3. Decision ===
        if prob >= PROB_THRESHOLD:
            print("✅ Signal confirmed. Executing long trade.")

            # === 4. Place market order ===
            order = place_market_order(TRADE_SYMBOL, "BUY", RISK_PER_TRADE, leverage=LEVERAGE)
            if order is None:
                print("[ERROR] Order not executed. Skipping...")
                time.sleep(60)
                continue

            # === 5. Extract filled price ===
            entry_price = float(order['fills'][0]['price']) if 'fills' in order else 0

            # === 6. Compute SL/TP ===
            stop_loss = entry_price * 0.98  # 2% risk
            take_profit = entry_price * 1.03  # 3% gain (1.5R logic)

            # === 7. Place SL/TP orders ===
            place_sl_tp_orders(
                symbol=TRADE_SYMBOL,
                position_side="LONG",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            # === 8. Log trade ===
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            log_row = f"{now},{TRADE_SYMBOL},{prob:.4f},LONG,{entry_price:.2f},{stop_loss:.2f},{take_profit:.2f}\n"
            with open(LOG_PATH, "a") as f:
                f.write(log_row)

        else:
            print("❌ No valid signal.")

    except Exception as e:
        print("[ERROR] Runtime failure:", e)

    time.sleep(60)  # wait 1 minute before next check
