
from __future__ import annotations

import logging
from typing import Optional

from execution.exchange.base import Exchange
from execution.portfolio import Portfolio
from execution.smart_router import SmartRouter

log = logging.getLogger("trade_manager")


class TradeManager:

    def __init__(self, router: SmartRouter):
        self.router = router

    # =========================================================
    # PARTIAL TAKE PROFIT
    # =========================================================

    async def place_partial_tp(
        self,
        ex: Exchange,
        symbol: str,
        qty: float,
        entry_price: float,
        tp_pct: float = 0.01
    ) -> Optional[object]:

        if qty <= 0:
            return None

        tp_price = entry_price * (1 + tp_pct)

        try:
            return await self.router.place_partial_tp_limit(
                ex,
                symbol,
                qty * 0.5,
                tp_price
            )

        except Exception as e:
            log.warning(
                "partial_tp_failed",
                extra={
                    "symbol": symbol,
                    "err": str(e)
                }
            )
            return None

    # =========================================================
    # SAFE OCO PLACEMENT
    # =========================================================

    async def place_safe_oco(
        self,
        ex: Exchange,
        symbol: str,
        qty: float,
        entry_price: float,
        tp_pct: float,
        sl_pct: float
    ) -> bool:

        try:
            tp, sl = await self.router.place_oco_tp_sl(
                ex,
                symbol,
                qty,
                entry_price,
                tp_pct,
                sl_pct
            )

            if not tp or not sl:
                log.warning(
                    "OCO_PLACE_FAILED",
                    extra={"symbol": symbol}
                )
                return False

            ok = await self.router.verify_oco(ex, symbol)

            if not ok:
                log.warning(
                    "OCO_VERIFY_FAILED",
                    extra={"symbol": symbol}
                )

            return ok

        except Exception as e:
            log.warning(
                "place_safe_oco_error",
                extra={
                    "symbol": symbol,
                    "err": str(e)
                }
            )
            return False

    # =========================================================
    # CANCEL ALL ORDERS (SAFE CLEANUP)
    # =========================================================

    async def cancel_all_orders(
        self,
        ex: Exchange,
        symbol: str
    ) -> None:

        try:
            await ex.cancel_all(symbol)

            log.info(
                "cancel_all_orders",
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

    # =========================================================
    # CLOSE POSITION SAFELY
    # =========================================================

    async def close_position(
        self,
        ex: Exchange,
        portfolio: Portfolio,
        symbol: str
    ) -> None:

        if not portfolio.has_position(symbol):
            return

        pos = portfolio.positions.get(symbol)

        if not pos:
            return

        qty = pos.qty

        try:
            await self.cancel_all_orders(ex, symbol)

            await self.router.close_long(
                ex,
                symbol,
                qty
            )

            portfolio.close_position(symbol)

            log.info(
                "position_closed",
                extra={
                    "symbol": symbol,
                    "qty": qty
                }
            )

        except Exception as e:
            log.warning(
                "close_position_failed",
                extra={
                    "symbol": symbol,
                    "err": str(e)
                }
            )

    # =========================================================
    # EMERGENCY CLOSE (LAST RESORT)
    # =========================================================

    async def emergency_close(
        self,
        ex: Exchange,
        portfolio: Portfolio,
        symbol: str
    ) -> None:

        if not portfolio.has_position(symbol):
            return

        pos = portfolio.positions.get(symbol)

        if not pos:
            return

        try:
            await self.router.close_position(
                ex,
                symbol,
                pos.qty
            )

            portfolio.close_position(symbol)

            log.critical(
                "EMERGENCY_POSITION_CLOSE",
                extra={
                    "symbol": symbol,
                    "qty": pos.qty
                }
            )

        except Exception as e:
            log.critical(
                "EMERGENCY_CLOSE_FAILED",
                extra={
                    "symbol": symbol,
                    "err": str(e)
                }
            )
