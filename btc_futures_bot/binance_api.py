# binance_api.py

import time
import requests
import hmac
import hashlib
import os
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
BASE_URL = "https://fapi.binance.com"

HEADERS = {
    "X-MBX-APIKEY": API_KEY
}

# === Utility ===
def _get_timestamp():
    return int(time.time() * 1000)

def _sign(params):
    query = urlencode(params)
    return hmac.new(API_SECRET.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()

def _request(method, endpoint, params=None):
    if params is None:
        params = {}
    params['timestamp'] = _get_timestamp()
    query = urlencode(params)
    signature = _sign(params)
    url = f"{BASE_URL}{endpoint}?{query}&signature={signature}"

    try:
        if method == "GET":
            response = requests.get(url, headers=HEADERS)
        elif method == "POST":
            response = requests.post(url, headers=HEADERS)
        else:
            raise ValueError("Invalid HTTP method")

        response.raise_for_status()
        return response.json()
    except Exception as e:
        print("[ERROR] Binance API request failed:", e)
        return None

# === Market Data ===
def get_klines(symbol="BTCUSDT", interval="1m", limit=1000):
    try:
        url = f"{BASE_URL}/fapi/v1/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        response = requests.get(url, params=params, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch klines for {symbol}:", e)
        return []

# === Account Info ===
def get_account_balance():
    result = _request("GET", "/fapi/v2/account")
    if result:
        usdt_balance = next((a for a in result['assets'] if a['asset'] == 'USDT'), None)
        if usdt_balance:
            return float(usdt_balance['availableBalance'])
    return 0

# === Order Placement ===
def place_market_order(symbol, side, usdt_amount, leverage=10):
    # Set leverage first (isolated)
    _request("POST", "/fapi/v1/leverage", {
        "symbol": symbol.upper(),
        "leverage": leverage
    })

    # Get current price to compute quantity
    price_data = get_klines(symbol, "1m", limit=1)
    if not price_data:
        print("[ERROR] Could not fetch price for quantity calc")
        return None

    mark_price = float(price_data[-1][4])  # closing price of last candle
    quantity = round(usdt_amount / mark_price, 3)  # round to 3 decimal places

    params = {
        "symbol": symbol.upper(),
        "side": side.upper(),
        "type": "MARKET",
        "quantity": quantity
    }

    result = _request("POST", "/fapi/v1/order", params)
    if result:
        print(f"[ORDER] Placed {side} {quantity} {symbol} @ ${mark_price:.2f}")
    return result

# === Example (remove when in production) ===
if __name__ == "__main__":
    # Example test run (make sure you're on testnet if you're testing)
    bal = get_account_balance()
    print(f"Available USDT Balance: {bal}")

    # place_market_order("BTCUSDT", "BUY", usdt_amount=10)
    pass
