from __future__ import annotations

import logging

log = logging.getLogger("position_manager")


class PositionManager:

    def __init__(
        self,
        tp_pct: float = 0.02,
        sl_pct: float = 0.01,
        max_bars: int = 30,
    ):

        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.max_bars = max_bars

    async def maybe_close_position(
        self,
        router,
        exchange,
        portfolio,
        symbol: str,
        price: float,
        bar_index: int,
    ) -> None:

        # =========================
        # POSITION CHECK
        # =========================

        if not portfolio.has_position(symbol):
            return

        pos = portfolio.get(symbol)

        if pos is None:
            return

        entry_price = pos.entry_price

        if entry_price is None or entry_price == 0:
            return

        pnl_pct = (price - entry_price) / entry_price

        # =========================
        # DEBUG LOG
        # =========================

        log.info(
            f"POSITION_DEBUG {symbol} "
            f"entry={entry_price:.6f} "
            f"price={price:.6f} "
            f"pnl={pnl_pct:.6f} "
            f"tp_distance={(self.tp_pct - pnl_pct):.6f} "
            f"sl_distance={(pnl_pct + self.sl_pct):.6f}"
        )

        # =========================
        # TAKE PROFIT
        # =========================

        if pnl_pct >= self.tp_pct:

            log.info(
                f"TP_TRIGGER {symbol} pnl={pnl_pct:.4f}"
            )

            order = await router.close_long(
                exchange,
                symbol,
                pos.qty
            )

            if not order:
                log.error(
                    f"SELL_FAILED {symbol}"
                )
                return

            portfolio.close(symbol)

            log.info(
                f"SELL_EXECUTED {symbol}"
            )

            return

        # =========================
        # STOP LOSS
        # =========================

        if pnl_pct <= -self.sl_pct:

            log.info(
                f"SL_TRIGGER {symbol} pnl={pnl_pct:.4f}"
            )

            order = await router.close_long(
                exchange,
                symbol,
                pos.qty
            )

            if not order:
                log.error(
                    f"SELL_FAILED {symbol}"
                )
                return

            portfolio.close(symbol)

            log.info(
                f"SELL_EXECUTED {symbol}"
            )

            return

        # =========================
        # TIME EXIT
        # =========================

        try:
            entry_bar = getattr(pos, "entry_index", 0)
        except Exception:
            entry_bar = 0

        if (bar_index - entry_bar) >= self.max_bars:

            log.info(
                f"TIME_EXIT {symbol}"
            )

            order = await router.close_long(
                exchange,
                symbol,
                pos.qty
            )

            if not order:
                log.error(
                    f"SELL_FAILED {symbol}"
                )
                return

            portfolio.close(symbol)

            log.info(
                f"SELL_EXECUTED {symbol}"
            )
