from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from execution.exchange.base import Exchange, OrderResult

log = logging.getLogger("smart_router")


@dataclass
class ExecResult:
    entry: Optional[OrderResult] = None
    partial_tp_order: Optional[OrderResult] = None
    exit: Optional[OrderResult] = None


class SmartRouter:

    async def open_long(self, ex: Exchange, symbol: str, quote_usdt: float) -> Optional[OrderResult]:

        log.info(
            "open_long_request",
            extra={
                "exchange": ex.name,
                "symbol": symbol,
                "quote_usdt": quote_usdt
            }
        )

        balance = await ex.get_usdt_balance()

        if balance < quote_usdt:

            log.info(
                "SKIP_TRADE_LOW_BALANCE",
                extra={
                    "exchange": ex.name,
                    "symbol": symbol,
                    "balance": balance,
                    "required": quote_usdt
                }
            )

            return None

        res = await ex.market_buy_quote(symbol, quote_usdt)

        log.info(
            "open_long_done",
            extra={
                "exchange": ex.name,
                "symbol": symbol,
                "qty": res.get("qty") if isinstance(res, dict) else getattr(res, "executed_qty", None),
                "avg": res.get("avg_price") if isinstance(res, dict) else getattr(res, "avg_price", None),
                "status": res.get("status") if isinstance(res, dict) else getattr(res, "status", None)
            }
        )

        return res


    async def place_partial_tp_limit(
        self,
        ex: Exchange,
        symbol: str,
        qty: float,
        tp_price: float
    ) -> Optional[OrderResult]:

        try:

            o = await ex.limit_sell_base(symbol, qty, tp_price)

            log.info(
                "partial_tp_limit_placed",
                extra={
                    "exchange": ex.name,
                    "symbol": symbol,
                    "qty": qty,
                    "tp": tp_price,
                    "order_id": o.order_id
                }
            )

            return o

        except Exception as e:

            log.warning(
                "partial_tp_limit_failed",
                extra={
                    "exchange": ex.name,
                    "symbol": symbol,
                    "err": str(e)
                }
            )

            return None


    async def close_long_market(
        self,
        ex: Exchange,
        symbol: str,
        qty: float
    ) -> Optional[OrderResult]:

        log.info(
            "close_long_request",
            extra={
                "exchange": ex.name,
                "symbol": symbol,
                "qty": qty
            }
        )

        res = await ex.market_sell_base(symbol, qty)

        log.info(
            "close_long_done",
            extra={
                "exchange": ex.name,
                "symbol": symbol,
                "qty": getattr(res, "executed_qty", None),
                "avg": getattr(res, "avg_price", None),
                "status": getattr(res, "status", None)
            }
        )

        return res


    # =========================================================
    # WRAPPER FOR POSITION MANAGER
    # =========================================================
    async def close_long(
        self,
        ex: Exchange,
        symbol: str,
        qty: float
    ) -> Optional[OrderResult]:

        if qty <= 0:
            return None

        return await self.close_long_market(ex, symbol, qty)


    # =========================================================
    # OCO TAKE PROFIT + STOP LOSS
    # =========================================================
    async def place_oco_tp_sl(
        self,
        ex: Exchange,
        symbol: str,
        qty: float,
        entry_price: float,
        tp_pct: float = 0.02,
        sl_pct: float = 0.01
    ):

        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)

        log.info(
            "placing_oco_orders",
            extra={
                "symbol": symbol,
                "tp": tp_price,
                "sl": sl_price,
                "qty": qty
            }
        )

        try:

            tp_order = await ex.limit_sell_base(symbol, qty, tp_price)

            sl_order = await ex.stop_market_sell(
                symbol,
                qty,
                sl_price
            )

            log.info(
                "oco_orders_placed",
                extra={
                    "symbol": symbol,
                    "tp_order": getattr(tp_order, "order_id", None),
                    "sl_order": getattr(sl_order, "order_id", None)
                }
            )

            return tp_order, sl_order

        except Exception as e:

            log.warning(
                "oco_order_failed",
                extra={"symbol": symbol, "err": str(e)}
            )

            return None, None


    # =========================================================
    # VERIFY OCO EXISTS (used by main.py)
    # =========================================================
    async def verify_oco(self, ex: Exchange, symbol: str) -> bool:

        try:

            orders = await ex.fetch_open_orders(symbol)

            tp_found = False
            sl_found = False

            for o in orders:

                otype = str(o.get("type", "")).lower()

                if "limit" in otype or "take_profit" in otype:
                    tp_found = True

                if "stop" in otype:
                    sl_found = True

            log.info(
                "verify_oco_result",
                extra={
                    "symbol": symbol,
                    "tp_found": tp_found,
                    "sl_found": sl_found
                }
            )

            return tp_found and sl_found

        except Exception as e:

            log.warning(
                "verify_oco_failed",
                extra={
                    "symbol": symbol,
                    "err": str(e)
                }
            )

            return False


    # =========================================================
    # EMERGENCY CLOSE (OCO failed protection)
    # =========================================================
    async def close_position(
        self,
        ex: Exchange,
        symbol: str,
        qty: float
    ) -> Optional[OrderResult]:

        if qty <= 0:
            return None

        try:

            log.critical(
                "EMERGENCY_CLOSE_POSITION",
                extra={
                    "exchange": ex.name,
                    "symbol": symbol,
                    "qty": qty
                }
            )

            res = await ex.market_sell_base(symbol, qty)

            log.critical(
                "EMERGENCY_CLOSE_DONE",
                extra={
                    "exchange": ex.name,
                    "symbol": symbol,
                    "qty": getattr(res, "executed_qty", None),
                    "avg": getattr(res, "avg_price", None)
                }
            )

            return res

        except Exception as e:

            log.critical(
                "EMERGENCY_CLOSE_FAILED",
                extra={
                    "symbol": symbol,
                    "err": str(e)
                }
            )

            return None


    async def cancel_all(self, ex: Exchange, symbol: str) -> None:

        try:

            await ex.cancel_all(symbol)

            log.info(
                "cancel_all_ok",
                extra={
                    "exchange": ex.name,
                    "symbol": symbol
                }
            )

        except Exception as e:

            log.warning(
                "cancel_all_failed",
                extra={
                    "exchange": ex.name,
                    "symbol": symbol,
                    "err": str(e)
                }
            )
