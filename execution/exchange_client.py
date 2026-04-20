import os
import time
import logging
from typing import Any, Dict, Optional

import ccxt

logger = logging.getLogger("gbm")


class ExchangeClientError(Exception):
    pass


class LiveTradingBlocked(Exception):
    pass


class BinanceSpotClient:
    """
    Binance Spot client.
    """

    def __init__(self):
        self.mode = os.getenv("MODE", "DEMO").upper()  # DEMO | TESTNET | LIVE
        self.kill_switch = os.getenv("KILL_SWITCH", "false").lower() == "true"
        self.live_confirmation = os.getenv("LIVE_CONFIRMATION", "false").lower() == "true"

        self.max_quote_per_trade = float(os.getenv("MAX_QUOTE_PER_TRADE", "10"))
        self.symbol_whitelist = set(
            s.strip().upper()
            for s in os.getenv("SYMBOL_WHITELIST", "BTC/USDT").split(",")
            if s.strip()
        )

        self.order_retry_count    = int(os.getenv("ORDER_RETRY_COUNT",    "3"))
        self.order_retry_delay_ms = int(os.getenv("ORDER_RETRY_DELAY_MS", "400"))

        self.spread_limit_pct = float(
            os.getenv("SPREAD_LIMIT_PERCENT") or os.getenv("MAX_SPREAD_PCT") or "0.12"
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # BINANCE API KEYS (Render ENV-დან)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        api_key    = os.getenv("BINANCE_API_KEY",    "").strip()
        api_secret = os.getenv("BINANCE_API_SECRET", "").strip()

        if self.mode in ("LIVE", "TESTNET"):
            if not api_key or not api_secret:
                raise ExchangeClientError("Missing BINANCE_API_KEY / BINANCE_API_SECRET for LIVE/TESTNET.")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ccxt.binance — Spot რეჟიმი
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        exchange_config = {
            "apiKey":          api_key,
            "secret":          api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType":      "spot",   # Binance Spot
                "fetchCurrencies":  False,    # ზედმეტი API call
                "fetchTradingFees": False,    # ზედმეტი API call
            },
        }

        if self.mode == "TESTNET":
            exchange_config["urls"] = {
                "api": {
                    "public":  "https://testnet.binance.vision/api",
                    "private": "https://testnet.binance.vision/api",
                }
            }
            logger.info("BINANCE_TESTNET | testnet mode enabled")

        self.exchange = ccxt.binance(exchange_config)

        # load_markets ერთხელ startup-ზე — ccxt cache-ავს შედეგს.
        # fetch_ohlcv-ზე load_markets-ი აღარ გამოიძახება — rate limit fix.
        try:
            self.exchange.load_markets()
            logger.info("BINANCE_LOAD_MARKETS | OK | markets cached")
        except Exception as e:
            logger.warning(f"LOAD_MARKETS_WARN | err={e}")

    def _guard(self, symbol: str, quote_amount: Optional[float] = None) -> None:
        if self.kill_switch:
            raise LiveTradingBlocked("KILL_SWITCH is ON.")
        if self.mode == "LIVE" and not self.live_confirmation:
            raise LiveTradingBlocked("LIVE_CONFIRMATION is OFF.")
        if self.mode == "DEMO":
            raise LiveTradingBlocked("MODE=DEMO -> exchange client must not execute real orders.")
        if symbol and symbol.upper() not in self.symbol_whitelist:
            raise LiveTradingBlocked(f"Symbol not allowed by whitelist: {symbol}.")
        if quote_amount is not None and quote_amount > self.max_quote_per_trade:
            raise LiveTradingBlocked(f"quote_amount {quote_amount} exceeds MAX_QUOTE_PER_TRADE={self.max_quote_per_trade}")

    def _with_retry(self, fn, *args, label: str = "ORDER", **kwargs):
        """
        ORDER_RETRY_COUNT / ORDER_RETRY_DELAY_MS — exponential backoff.
        NetworkError / RequestTimeout → retry. სხვა exception → immediately raise.
        """
        delay_s  = self.order_retry_delay_ms / 1000.0
        last_err = None
        for attempt in range(1, self.order_retry_count + 1):
            try:
                return fn(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                last_err = e
                if attempt < self.order_retry_count:
                    wait = delay_s * (2 ** (attempt - 1))
                    logger.warning(
                        f"{label}_RETRY | attempt={attempt}/{self.order_retry_count} "
                        f"wait={wait:.2f}s err={e}"
                    )
                    time.sleep(wait)
            except Exception:
                raise
        raise ExchangeClientError(
            f"{label}_RETRY_EXHAUSTED after {self.order_retry_count} attempts | last_err={last_err}"
        )

    def diagnostics(self) -> Dict[str, Any]:
        try:
            sym = next(iter(self.symbol_whitelist)) if self.symbol_whitelist else "BTC/USDT"
            t = self.exchange.fetch_ticker(sym)
            usdt_free = self.fetch_balance_free("USDT")
            return {
                "mode": self.mode,
                "kill_switch": self.kill_switch,
                "live_confirmation": self.live_confirmation,
                "symbol_probe": sym,
                "last_price": float(t.get("last") or 0.0),
                "usdt_free": usdt_free,
                "ok": True,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def fetch_last_price(self, symbol: str) -> float:
        t = self.exchange.fetch_ticker(symbol)
        return float(t["last"])

    def get_min_notional(self, symbol: str) -> float:
        """Binance Spot minimum notional for an order."""
        try:
            m = self.exchange.market(symbol)
            cost_min = (((m.get("limits") or {}).get("cost") or {}).get("min"))
            if cost_min is not None:
                return float(cost_min)
            for f in (m.get("info") or {}).get("filters", []):
                if f.get("filterType") in ("MIN_NOTIONAL", "NOTIONAL"):
                    return float(f.get("minNotional", 0.0))
        except Exception as e:
            logger.warning(f"MIN_NOTIONAL_LOOKUP_FAIL | symbol={symbol} err={e}")
        return 10.0  # Binance Spot default $10

    def fetch_balance_free(self, asset: str) -> float:
        # Binance Spot — type=spot კმარა, unified Bybit-ისთვის იყო
        try:
            bal = self.exchange.fetch_balance()
            return float((bal.get("free", {}) or {}).get(asset.upper(), 0.0) or 0.0)
        except Exception as e:
            logger.warning(f"FETCH_BALANCE_FAIL | asset={asset} err={e}")
            return 0.0

    def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        return self.exchange.fetch_order(str(order_id), symbol)

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        return self.exchange.cancel_order(str(order_id), symbol)

    # ----------------------------
    # Precision helpers (STRING!)
    # ----------------------------
    def floor_amount(self, symbol: str, amount: float) -> float:
        try:
            s = self.exchange.amount_to_precision(symbol, amount)
            return float(s)
        except Exception:
            return float(amount)

    def floor_price(self, symbol: str, price: float) -> float:
        try:
            s = self.exchange.price_to_precision(symbol, price)
            return float(s)
        except Exception:
            return float(price)

    def _amount_str(self, symbol: str, amount: float) -> str:
        return str(self.exchange.amount_to_precision(symbol, amount))

    def _price_str(self, symbol: str, price: float) -> str:
        return str(self.exchange.price_to_precision(symbol, price))

    # ----------------------------
    # Orders
    # ----------------------------
    def place_market_buy_by_quote(self, symbol: str, quote_amount: float) -> Dict[str, Any]:
        """
        Bybit Spot market buy by quote (USDT).
        Bybit ccxt-ში: create_order(..., params={"quoteOrderQty": ...}) ან
        createMarketBuyOrderWithCost — ccxt unified method.
        """
        self._guard(symbol, quote_amount=quote_amount)
        try:
            # ccxt unified: cost-based market buy
            return self._with_retry(
                self.exchange.create_order,
                symbol, "market", "buy", None, None,
                {"quoteOrderQty": float(quote_amount)},
                label="MARKET_BUY"
            )
        except ExchangeClientError:
            raise
        except Exception as e:
            raise ExchangeClientError(f"Market buy failed: {e}")

    def place_market_sell(self, symbol: str, base_amount: float) -> Dict[str, Any]:
        self._guard(symbol)
        try:
            amt = float(self.exchange.amount_to_precision(symbol, base_amount))
            return self._with_retry(
                self.exchange.create_order, symbol, "market", "sell", float(amt), None,
                label="MARKET_SELL"
            )
        except ExchangeClientError:
            raise
        except Exception as e:
            raise ExchangeClientError(f"Market sell failed: {e}")

    def place_limit_sell_amount(self, symbol: str, base_amount: float, price: float) -> Dict[str, Any]:
        self._guard(symbol)
        try:
            amt = float(self.exchange.amount_to_precision(symbol, base_amount))
            px  = float(self.exchange.price_to_precision(symbol, price))
            return self._with_retry(
                self.exchange.create_order, symbol, "limit", "sell", float(amt), float(px),
                label="LIMIT_SELL"
            )
        except ExchangeClientError:
            raise
        except Exception as e:
            raise ExchangeClientError(f"Limit sell failed: {e}")

    def place_stop_loss_limit_sell(self, symbol: str, base_amount: float, stop_price: float, limit_price: float) -> Dict[str, Any]:
        # Binance Spot STOP_LOSS_LIMIT order
        self._guard(symbol)
        try:
            amt      = float(self.exchange.amount_to_precision(symbol, base_amount))
            stop_px  = float(self.exchange.price_to_precision(symbol, stop_price))
            limit_px = float(self.exchange.price_to_precision(symbol, limit_price))
            params = {"stopPrice": stop_px, "timeInForce": "GTC"}
            return self._with_retry(
                self.exchange.create_order, symbol, "STOP_LOSS_LIMIT", "sell",
                float(amt), float(limit_px), params,
                label="SL_LIMIT_SELL"
            )
        except ExchangeClientError:
            raise
        except Exception as e:
            raise ExchangeClientError(f"Stop-loss-limit sell failed: {e}")


    def place_oco_sell(self, symbol: str, base_amount: float, tp_price: float, sl_stop_price: float, sl_limit_price: float) -> Dict[str, Any]:
        """
        Binance Spot native OCO sell order.
        TP (Limit) + SL (Stop-Limit) ერთ OCO ბრძანებაში.
        """
        self._guard(symbol)
        try:
            qty      = self._amount_str(symbol, base_amount)
            tp_px    = self._price_str(symbol, tp_price)
            sl_stop  = self._price_str(symbol, sl_stop_price)
            sl_limit = self._price_str(symbol, sl_limit_price)

            result = self._with_retry(
                self.exchange.create_order,
                symbol, "STOP_LOSS_LIMIT", "sell", float(qty), float(sl_limit),
                {
                    "stopPrice":      float(sl_stop),
                    "aboveType":      "LIMIT_MAKER",
                    "abovePrice":     float(tp_px),
                    "belowType":      "STOP_LOSS_LIMIT",
                    "belowStopPrice": float(sl_stop),
                    "belowPrice":     float(sl_limit),
                    "quantity":       float(qty),
                    "newOrderRespType": "FULL",
                },
                label="OCO_SELL"
            )

            return {"raw": result}
        except Exception as e:
            raise ExchangeClientError(f"OCO sell failed: {e}")
