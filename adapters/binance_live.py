
import ccxt

def create_exchange(api_key, api_secret):
    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
    })
    return exchange
