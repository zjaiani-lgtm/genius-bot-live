"""
Institutional-grade Bybit REST client (Spot V5)

Features
--------
• Safe signing
• HTTP protection
• Retry system
• Rate-limit handling
• Precision-safe orders
• Proper balance parsing
• Async session reuse
"""

import aiohttp
import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Dict, List, Any, Optional
from urllib.parse import urlencode


logger = logging.getLogger(__name__)


# ==========================================================
# INTERVAL NORMALIZATION
# ==========================================================

def _normalize_interval(interval: str) -> str:

    mapping = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D",
        "1w": "W",
        "1M": "M",
    }

    return mapping.get(interval, interval)


# ==========================================================
# BYBIT REST CLIENT
# ==========================================================

class BybitREST:

    BASE_URL = "https://api.bybit.com"

    MAX_RETRIES = 3
    RETRY_BACKOFF = 0.5

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        recv_window: int = 5000,
        timeout: int = 15,
    ):

        if not api_key or not api_secret:
            raise RuntimeError("Bybit API credentials missing")

        self.name = "bybit"
        self.api_key = api_key
        self.api_secret = api_secret
        self.recv_window = recv_window
        self.timeout = timeout

        self._session: Optional[aiohttp.ClientSession] = None
        self._symbol_precisions: Dict[str, int] = {}

    # ------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)

        return self._session

    # ------------------------------------------------------

    async def close(self):

        if self._session and not self._session.closed:
            await self._session.close()

    # ==========================================================
    # SIGNING
    # ==========================================================

    def _sign(self, timestamp: str, payload: str) -> str:

        sign_payload = (
            timestamp
            + self.api_key
            + str(self.recv_window)
            + payload
        )

        return hmac.new(
            self.api_secret.encode(),
            sign_payload.encode(),
            hashlib.sha256
        ).hexdigest()

    # ==========================================================
    # HTTP REQUEST WRAPPER
    # ==========================================================

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        body: Optional[Dict] = None,
        private: bool = False,
    ) -> Dict:

        url = f"{self.BASE_URL}{endpoint}"

        session = await self._get_session()

        for attempt in range(self.MAX_RETRIES):

            try:

                headers = {}

                query_string = urlencode(params) if params else ""
                body_string = json.dumps(body, separators=(",", ":")) if body else ""

                payload = query_string + body_string

                if private:

                    timestamp = str(int(time.time() * 1000))

                    signature = self._sign(timestamp, payload)

                    headers.update({
                        "X-BAPI-API-KEY": self.api_key,
                        "X-BAPI-TIMESTAMP": timestamp,
                        "X-BAPI-SIGN": signature,
                        "X-BAPI-RECV-WINDOW": str(self.recv_window),
                    })

                if body:
                    headers["Content-Type"] = "application/json"

                async with session.request(
                    method,
                    url,
                    params=params,
                    data=body_string if body else None,
                    headers=headers,
                ) as resp:

                    if resp.status != 200:

                        text = await resp.text()

                        raise RuntimeError(
                            f"HTTP_ERROR {resp.status} {text}"
                        )

                    data = await resp.json()

                ret = data.get("retCode")

                if ret == 0:
                    return data

                if ret in (10006, 10016):

                    logger.warning("BYBIT_RATE_LIMIT retrying...")

                    await asyncio.sleep(
                        self.RETRY_BACKOFF * (attempt + 1)
                    )

                    continue

                raise RuntimeError(f"BYBIT_API_ERROR {data}")

            except Exception:

                if attempt == self.MAX_RETRIES - 1:
                    raise

                await asyncio.sleep(
                    self.RETRY_BACKOFF * (attempt + 1)
                )

        raise RuntimeError("BYBIT_REQUEST_FAILED")

    # ==========================================================
    # BALANCES
    # ==========================================================

    async def fetch_balances(self) -> Dict[str, float]:

        data = await self._request(
            "GET",
            "/v5/account/wallet-balance",
            params={"accountType": "UNIFIED"},
            private=True,
        )

        balances: Dict[str, float] = {}

        try:

            coins = data["result"]["list"][0]["coin"]

            for c in coins:

                coin = c["coin"]

                available = float(
                    c.get("walletBalance")
                    or 0
                )

                balances[coin] = available

        except Exception as e:

            logger.error(f"BALANCE_PARSE_ERROR {e}")

        return balances


    async def fetch_usdt_balance(self) -> float:
        balances = await self.fetch_balances()
        return balances.get("USDT", 0.0)


    async def get_usdt_balance(self) -> float:
        balances = await self.fetch_balances()
        return balances.get("USDT", 0.0)


    # ==========================================================
    # FETCH OPEN ORDERS
    # ==========================================================

    async def fetch_open_orders(self, symbol: str):

        data = await self._request(
            "GET",
            "/v5/order/realtime",
            params={
                "category": "spot",
                "symbol": symbol,
            },
            private=True,
        )

        return data["result"]["list"]


    # ==========================================================
    # FETCH OHLCV
    # ==========================================================

    async def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:

        interval = _normalize_interval(interval)

        data = await self._request(
            "GET",
            "/v5/market/kline",
            params={
                "category": "spot",
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            },
        )

        raw = data["result"]["list"]

        raw.reverse()

        candles: List[Dict[str, Any]] = []

        for c in raw:

            candles.append({
                "ts": int(c[0]),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            })

        logger.info(
            f"FETCH_OHLCV_OK symbol={symbol} candles={len(candles)}"
        )

        return candles


    # ==========================================================
    # SYMBOL PRECISION
    # ==========================================================

    async def _get_symbol_precision(self, symbol: str) -> int:

        if symbol in self._symbol_precisions:
            return self._symbol_precisions[symbol]

        data = await self._request(
            "GET",
            "/v5/market/instruments-info",
            params={
                "category": "spot",
                "symbol": symbol
            },
        )

        instruments = data["result"]["list"]

        for inst in instruments:

            if inst["symbol"] == symbol:

                filters = inst["lotSizeFilter"]

                step = (
                    filters.get("quotePrecision")
                    or filters.get("basePrecision")
                )

                precision = len(step.split(".")[-1])

                self._symbol_precisions[symbol] = precision

                return precision

        return 6


    # ==========================================================
    # PRECISION SAFE QTY
    # ==========================================================

    async def _safe_qty(self, symbol: str, qty: float) -> str:

        precision = await self._get_symbol_precision(symbol)

        rounded = round(qty, precision)

        fmt = "{:0." + str(precision) + "f}"

        return fmt.format(rounded)


    # ==========================================================
    # MARKET BUY (QUOTE SIZE)
    # ==========================================================

    async def market_buy_quote(
        self,
        symbol: str,
        quote_amount: float,
    ) -> Dict[str, Any]:

        qty = await self._safe_qty(symbol, quote_amount)

        body = {
            "category": "spot",
            "symbol": symbol,
            "side": "Buy",
            "orderType": "Market",
            "qty": qty,
            "marketUnit": "quoteCoin",
        }

        data = await self._request(
            "POST",
            "/v5/order/create",
            body=body,
            private=True,
        )

        result = data.get("result", {})

        parsed = {
            "order_id": result.get("orderId"),
            "order_link_id": result.get("orderLinkId"),
            "symbol": symbol,
            "side": "Buy",
            "status": "submitted",
            "raw": result,
        }

        logger.info(f"MARKET_BUY_OK symbol={symbol}")

        return parsed


    # ==========================================================
    # MARKET SELL
    # ==========================================================

    async def market_sell(
        self,
        symbol: str,
        qty: float,
    ):

        qty = await self._safe_qty(symbol, qty)

        body = {
            "category": "spot",
            "symbol": symbol,
            "side": "Sell",
            "orderType": "Market",
            "qty": qty,
        }

        data = await self._request(
            "POST",
            "/v5/order/create",
            body=body,
            private=True,
        )

        result = data.get("result", {})

        logger.info(f"MARKET_SELL_OK symbol={symbol}")

        return result


BybitSpot = BybitREST
