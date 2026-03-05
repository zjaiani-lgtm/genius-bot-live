import logging
import aiohttp

log = logging.getLogger("bybit_test")


class BybitSpot:

    def __init__(self, base_url, api_key, api_secret, limiter):
        self.base_url = base_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.limiter = limiter

    async def market_buy_quote(self, symbol: str, quote_usdt: float):

        url = f"{self.base_url}/v5/order/create"

        payload = {
            "category": "spot",
            "symbol": symbol,
            "side": "Buy",
            "orderType": "Market",
            "qty": str(quote_usdt),
            "marketUnit": "quoteCoin"
        }

        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                data = await resp.json()

        # =========================================
        # RAW RESPONSE LOG (CRITICAL FOR DEBUG)
        # =========================================
        log.info(f"BYBIT_RAW_RESPONSE {data}")

        # =========================================
        # SAFE RESPONSE VALIDATION
        # =========================================

        if not isinstance(data, dict):
            raise Exception(f"Invalid Bybit response format: {data}")

        ret_code = data.get("retCode")

        if ret_code != 0:
            raise Exception(
                f"Bybit order failed retCode={ret_code} msg={data.get('retMsg')}"
            )

        result = data.get("result")
        if not result:
            raise Exception("Bybit response missing 'result' field")

        order_id = result.get("orderId") or result.get("order_id")

        if not order_id:
            raise Exception(f"Bybit response missing orderId field: {result}")

        # =========================================
        # UNIFIED ORDER OBJECT
        # =========================================

        class OrderResult:
            def __init__(self, order_id):
                self.order_id = str(order_id)
                self.executed_qty = None
                self.avg_price = None
                self.status = "SUBMITTED"

        return OrderResult(order_id)
