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
    TESTNET_REST_BASE = "https://testnet.binance.vision/api"

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

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ORDER_RETRY — transient Binance errors-ზე exponential backoff
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.order_retry_count    = int(os.getenv("ORDER_RETRY_COUNT",    "3"))
        self.order_retry_delay_ms = int(os.getenv("ORDER_RETRY_DELAY_MS", "400"))

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SPREAD_LIMIT_PERCENT — MAX_SPREAD_PCT alias (fallback chain)
        # execution_engine-ი MAX_SPREAD_PCT-ს კითხულობს, exchange_client
        # SPREAD_LIMIT_PERCENT-ს — ორივე ერთ ადგილობრივ slot-ს იზიარებს
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.spread_limit_pct = float(
            os.getenv("SPREAD_LIMIT_PERCENT") or os.getenv("MAX_SPREAD_PCT") or "0.12"
        )

        api_key    = os.getenv("BINANCE_API_KEY",    "").strip()
        api_secret = os.getenv("BINANCE_API_SECRET", "").strip()

        if self.mode in ("LIVE", "TESTNET"):
            if not api_key or not api_secret:
                raise ExchangeClientError("Missing BINANCE_API_KEY / BINANCE_API_SECRET for LIVE/TESTNET.")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # BINANCE_LIVE_REST_BASE — custom REST endpoint (regional / proxy)
        # default: Binance global. Override: https://api.binance.com/api/v3
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        live_rest_base = os.getenv("BINANCE_LIVE_REST_BASE", "").strip()

        self.exchange = ccxt.binance({
            "apiKey":          api_key,
            "secret":          api_secret,
            "enableRateLimit": True,
            "options":         {"defaultType": "spot"},
        })

        if self.mode == "TESTNET":
            self.exchange.urls["api"] = {
                "public":  self.TESTNET_REST_BASE,
                "private": self.TESTNET_REST_BASE,
            }
            self.exchange.options["fetchCurrencies"] = False
        elif self.mode == "LIVE" and live_rest_base:
            self.exchange.urls["api"]["private"] = live_rest_base
            self.exchange.urls["api"]["public"]  = live_rest_base
            logger.info(f"BINANCE_REST_OVERRIDE | base={live_rest_base}")

        # warm up markets for precision helpers
        try:
            self.exchange.load_markets()
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
        Delays: 0.4s → 0.8s → 1.6s (ORDER_RETRY_COUNT=3, DELAY_MS=400)
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
                raise  # non-transient — surface immediately
        raise ExchangeClientError(
            f"{label}_RETRY_EXHAUSTED after {self.order_retry_count} attempts | last_err={last_err}"
        )

    def diagnostics(self) -> Dict[str, Any]:
        try:
            bal = self.exchange.fetch_balance()
            sym = next(iter(self.symbol_whitelist)) if self.symbol_whitelist else "BTC/USDT"
            t = self.exchange.fetch_ticker(sym)
            return {
                "mode": self.mode,
                "kill_switch": self.kill_switch,
                "live_confirmation": self.live_confirmation,
                "symbol_probe": sym,
                "last_price": float(t.get("last") or 0.0),
                "usdt_free": float((bal.get("free", {}) or {}).get("USDT", 0.0) or 0.0),
                "ok": True,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def fetch_last_price(self, symbol: str) -> float:
        t = self.exchange.fetch_ticker(symbol)
        return float(t["last"])

    def get_min_notional(self, symbol: str) -> float:
        """Return minimum notional (quote value) required for an order on this symbol.

        Binance may reject market orders if the quote value is below MIN_NOTIONAL/NOTIONAL filter.
        We try multiple sources (ccxt limits then raw exchange filters) and return 0.0 if unknown.
        """
        try:
            m = self.exchange.market(symbol)

            # 1) ccxt normalized limits (if available)
            cost_min = (((m.get("limits") or {}).get("cost") or {}).get("min"))
            if cost_min is not None:
                return float(cost_min)

            # 2) raw Binance filters
            info = m.get("info") or {}
            filters = info.get("filters") or []
            for f in filters:
                t = str(f.get("filterType") or "").upper()
                if t in ("MIN_NOTIONAL", "NOTIONAL"):
                    v = f.get("minNotional")
                    if v is None:
                        v = f.get("minNotionalValue")
                    if v is None:
                        v = f.get("notional")
                    if v is not None:
                        return float(v)
        except Exception as e:
            logger.warning(f"MIN_NOTIONAL_LOOKUP_FAIL | symbol={symbol} err={e}")

        return 0.0

    def fetch_balance_free(self, asset: str) -> float:
        bal = self.exchange.fetch_balance()
        return float((bal.get("free", {}) or {}).get(asset.upper(), 0.0) or 0.0)

    def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        return self.exchange.fetch_order(str(order_id), symbol)

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        return self.exchange.cancel_order(str(order_id), symbol)

    # ----------------------------
    # Precision helpers (STRING!)
    # ----------------------------
    def floor_amount(self, symbol: str, amount: float) -> float:
        """
        Returns float but derived from amount_to_precision (string) to avoid float artifacts.
        """
        try:
            s = self.exchange.amount_to_precision(symbol, amount)  # string like "0.00018"
            return float(s)
        except Exception:
            return float(amount)

    def floor_price(self, symbol: str, price: float) -> float:
        """
        Returns float but derived from price_to_precision (string) to avoid float artifacts.
        """
        try:
            s = self.exchange.price_to_precision(symbol, price)  # string like "76253.90"
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
        self._guard(symbol, quote_amount=quote_amount)
        try:
            params = {"quoteOrderQty": float(quote_amount)}
            return self._with_retry(
                self.exchange.create_order, symbol, "market", "buy", None, None, params,
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
        self._guard(symbol)
        try:
            amt      = float(self.exchange.amount_to_precision(symbol, base_amount))
            stop_px  = float(self.exchange.price_to_precision(symbol, stop_price))
            limit_px = float(self.exchange.price_to_precision(symbol, limit_price))
            params   = {"stopPrice": stop_px, "timeInForce": "GTC"}
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
        Native Binance Spot OCO (single reserve).
        IMPORTANT: use STRING precision to avoid -1111 price precision errors.
        """
        self._guard(symbol)
        try:
            qty = self._amount_str(symbol, base_amount)
            price = self._price_str(symbol, tp_price)
            stop_price = self._price_str(symbol, sl_stop_price)
            stop_limit_price = self._price_str(symbol, sl_limit_price)

            payload = {
                "symbol": self.exchange.market_id(symbol),
                "side": "SELL",
                "quantity": qty,
                "price": price,
                "stopPrice": stop_price,
                "stopLimitPrice": stop_limit_price,
                "stopLimitTimeInForce": "GTC",
            }

            # direct endpoint call (stable)
            res = self.exchange.privatePostOrderOco(payload)
            return {"raw": res}
        except Exception as e:
            raise ExchangeClientError(f"OCO sell failed: {e}")
